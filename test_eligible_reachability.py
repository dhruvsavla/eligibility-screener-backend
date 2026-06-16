"""
Screens DIABETIC synthetic patients against the flagship protocol (REWIND)
to determine whether a fitting patient can reach ELIGIBLE.

If the DB is empty (no real patients), generates 20 in-memory patients using
the eligible/ineligible profiles from SyntheticPatientGenerator and filters to
diabetic ones — the exact population that *should* qualify for a T2D trial.
"""
import asyncio
import json
from app import database
from app.models.protocol import CriterionRule, CriterionType
from app.services.fhir_parser import FHIRParser
from app.services.scoring_engine import ScoringEngine
from app.services.synthea_generator import SyntheticPatientGenerator


async def test():
    proto = await database.fetch_one("SELECT * FROM protocols WHERE is_flagship = 1")
    if not proto:
        print("ERROR: No flagship protocol found. Run /api/protocols/seed-all first.")
        return

    print(f"Flagship: id={proto['id']}  {proto['title']}\n")

    rule_rows = await database.fetch_all(
        "SELECT * FROM criterion_rules WHERE protocol_id = ?", (proto["id"],)
    )
    rules = [
        CriterionRule(
            id=r["id"],
            protocol_id=r["protocol_id"],
            criterion_text=r["criterion_text"],
            concept=r["concept"],
            operator=r["operator"],
            value=r["value"],
            required=bool(r["required"]),
            criterion_type=CriterionType(r["criterion_type"]),
            confidence=r["confidence"],
        )
        for r in rule_rows
    ]

    inclusion_rules = [r for r in rules if r.criterion_type == CriterionType.inclusion]
    print(f"{len(inclusion_rules)} inclusion criteria in flagship:")
    for r in inclusion_rules:
        print(f"    [{r.operator:15s}] {r.concept}")
    print()

    parser = FHIRParser()
    engine = ScoringEngine()
    patient_data_list = []

    # --- Attempt 1: pull from DB ---
    db_rows = await database.fetch_all(
        "SELECT * FROM patients WHERE is_ground_truth = 0 "
        "AND (fhir_json LIKE '%diabetes%' OR fhir_json LIKE '%Diabetes%' "
        "     OR fhir_json LIKE '%44054006%') "
        "LIMIT 10"
    )
    if db_rows:
        print(f"Found {len(db_rows)} diabetic patients in DB.\n")
        for row in db_rows:
            pd = parser.parse_bundle(json.loads(row["fhir_json"]))
            patient_data_list.append(pd)
    else:
        print("DB is empty — generating 20 synthetic patients in memory.\n")
        gen = SyntheticPatientGenerator()
        bundles = gen.generate_patients(20, seed=1234)
        for bundle in bundles:
            pd = parser.parse_bundle(bundle)
            if any("diabetes" in c.lower() for c in pd.conditions):
                patient_data_list.append(pd)
        print(f"Using {len(patient_data_list)} diabetic patients from generator.\n")

    print("=" * 70)
    counts: dict[str, int] = {"ELIGIBLE": 0, "REVIEW_NEEDED": 0, "INELIGIBLE": 0}

    for pd in patient_data_list:
        result = engine.evaluate_patient(pd, rules)
        verdict = result.overall_verdict.value
        counts[verdict] = counts.get(verdict, 0) + 1
        sb = result.score_breakdown

        print(
            f"\n{pd.patient_id}: {verdict}  "
            f"(score={result.fit_score}, "
            f"inclusion_pass_rate={sb['inclusion_pass_rate']:.0%}  "
            f"{sb['inclusion_pass']}/{sb['inclusion_total']} pass)"
        )

        incl_evals = [
            e for e in result.evaluations
            if str(getattr(e.criterion_type, "value", e.criterion_type)) == "inclusion"
        ]
        for e in incl_evals:
            status = str(getattr(e.status, "value", e.status))
            mark = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "?")
            print(f"    {mark} [{status:9s}] {e.concept}")

    print("\n" + "=" * 70)
    print(f"\nVERDICT DISTRIBUTION (diabetic patients): {counts}")
    print()

    if counts.get("ELIGIBLE", 0) > 0:
        print("✓ DIAGNOSIS: Fitting patients CAN reach ELIGIBLE.")
        print("  The random-sample 0-ELIGIBLE was a SAMPLING ARTIFACT.")
        print("  → NO FIX NEEDED. The system works correctly.")
    else:
        print("✗ DIAGNOSIS: Even fitting diabetic patients cannot reach ELIGIBLE.")
        print()
        print("  Look at the '?' AMBIGUOUS inclusion criteria above — those that are")
        print("  AMBIGUOUS on EVERY patient are inherently unverifiable from FHIR.")
        print("  They should be excluded from the inclusion_pass_rate denominator.")
        print()
        print("  → APPLY THE PHASE 2 FIX.")


if __name__ == "__main__":
    asyncio.run(test())
