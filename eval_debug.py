"""Debug one truly-ELIGIBLE ground-truth patient to see why it doesn't reach ELIGIBLE."""
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
    gt_rows = await database.fetch_all(
        "SELECT patient_id, fhir_json, ground_truth_verdict FROM patients "
        "WHERE is_ground_truth = 1 AND ground_truth_protocol_id = 16 AND ground_truth_verdict = 'ELIGIBLE' "
        "LIMIT 2"
    )
    parser = FHIRParser()
    engine = ScoringEngine()

    for row in gt_rows:
        pd = parser.parse_bundle(json.loads(row["fhir_json"]))
        result = engine.evaluate_patient(pd, rules)
        sb = result.score_breakdown
        print(f"\nPatient: {row['patient_id']}  truth={row['ground_truth_verdict']}")
        print(f"  Score:            {result.fit_score}  (need >= 75)")
        print(f"  Verdict:          {result.overall_verdict.value}")
        print(f"  incl_pass_rate:   {sb['inclusion_pass_rate']:.0%}  (need >= 60%)")
        print(f"  incl_pass/fail/ambig: {sb['inclusion_pass']}/{sb['inclusion_fail']}/{sb['inclusion_ambiguous']}")
        print(f"  excl_triggered:   {sb['exclusion_triggered']}  (must be 0)")
        print(f"  excl_ambig/clear: {sb['exclusion_ambiguous']}/{sb['exclusion_clear']}")
        print(f"  deductions:       {sb['deductions']}")
        print(f"  Inclusion failures:")
        for e in result.evaluations:
            t = str(getattr(e.criterion_type, "value", e.criterion_type))
            s = str(getattr(e.status, "value", e.status))
            if t == "inclusion" and s in ("FAIL", "AMBIGUOUS"):
                print(f"    [{s}] {e.concept}: {e.explanation}")

asyncio.run(run())
