"""Screen real (non-ground-truth) patients against the flagship, print verdict spread."""
import asyncio, json
from app import database
from app.services.scoring_engine import ScoringEngine
from app.models.protocol import CriterionRule, CriterionType
from app.services.fhir_parser import FHIRParser

async def run():
    rows = await database.fetch_all("SELECT * FROM criterion_rules WHERE protocol_id = 16")
    rules = [
        CriterionRule(
            id=r["id"], protocol_id=r["protocol_id"], criterion_text=r["criterion_text"],
            concept=r["concept"], operator=r["operator"], value=r["value"],
            required=bool(r["required"]), criterion_type=CriterionType(r["criterion_type"]),
            confidence=r["confidence"],
        )
        for r in rows
    ]
    patients_rows = await database.fetch_all(
        "SELECT patient_id, fhir_json FROM patients WHERE is_ground_truth = 0 LIMIT 30"
    )
    parser = FHIRParser()
    engine = ScoringEngine()
    counts = {"ELIGIBLE": 0, "REVIEW_NEEDED": 0, "INELIGIBLE": 0}

    # For diagnosing inclusion matching, track which inclusions go ambiguous
    incl_ambig_counts: dict[str, int] = {}

    for row in patients_rows:
        pd = parser.parse_bundle(json.loads(row["fhir_json"]))
        result = engine.evaluate_patient(pd, rules)
        verdict = result.overall_verdict.value
        counts[verdict] += 1
        sb = result.score_breakdown
        print(f"  {row['patient_id']}: {verdict} "
              f"(score={result.fit_score} pass_rate={sb['inclusion_pass_rate']:.0%} "
              f"excl_ambig={sb['exclusion_ambiguous']})")
        for e in result.evaluations:
            t = str(getattr(e.criterion_type, "value", e.criterion_type))
            s = str(getattr(e.status, "value", e.status))
            if t == "inclusion" and s == "AMBIGUOUS":
                incl_ambig_counts[e.concept] = incl_ambig_counts.get(e.concept, 0) + 1

    total = len(patients_rows)
    print(f"\nVerdict distribution over {total} real patients: {counts}")
    if counts["ELIGIBLE"] == 0:
        print("\n⚠ Still no ELIGIBLE — inclusion_pass_rate is the blocker.")
        print("Inclusion criteria going AMBIGUOUS most often (concept-matching gap):")
        for concept, n in sorted(incl_ambig_counts.items(), key=lambda x: -x[1]):
            print(f"  {n:3d}x  {concept}")
    else:
        print(f"\n✓ {counts['ELIGIBLE']} patients now reach ELIGIBLE")

asyncio.run(run())
