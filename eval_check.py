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
        "SELECT patient_id, fhir_json, ground_truth_verdict, failure_mode "
        "FROM patients WHERE is_ground_truth = 1 AND ground_truth_protocol_id = 16"
    )
    parser = FHIRParser()
    engine = ScoringEngine()
    tp = fp = tn = fn = 0
    specificity_breaks = []
    verdicts = {"ELIGIBLE": 0, "REVIEW_NEEDED": 0, "INELIGIBLE": 0}

    for row in gt_rows:
        pd = parser.parse_bundle(json.loads(row["fhir_json"]))
        result = engine.evaluate_patient(pd, rules)
        predicted = result.overall_verdict.value
        truth = row["ground_truth_verdict"]
        verdicts[predicted] += 1
        is_truly_eligible = truth in ("ELIGIBLE", "BORDERLINE")
        pred_pos = predicted in ("ELIGIBLE", "REVIEW_NEEDED")
        if is_truly_eligible and pred_pos:       tp += 1
        elif is_truly_eligible and not pred_pos: fn += 1
        elif not is_truly_eligible and pred_pos: fp += 1
        else:                                    tn += 1
        if truth == "INELIGIBLE" and predicted == "ELIGIBLE":
            specificity_breaks.append({
                "patient_id": row["patient_id"], "score": result.fit_score,
                "failure_mode": row["failure_mode"],
                "excl_triggered": result.score_breakdown["exclusion_triggered"],
                "excl_ambig": result.score_breakdown["exclusion_ambiguous"],
            })

    total = len(gt_rows)
    sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

    print("=" * 60)
    print("GROUND-TRUTH EVALUATION — protocol 16 (REWIND/ELIGIBLE fix)")
    print("=" * 60)
    print(f"Total patients: {total}")
    print(f"Sensitivity:    {sens:.1f}%  (TP={tp} FN={fn})  — must be >= 85%")
    print(f"Specificity:    {spec:.1f}%  (TN={tn} FP={fp})")
    print(f"Verdict spread: {verdicts}")
    print(f"Specificity breaks (INELIGIBLE → ELIGIBLE): {len(specificity_breaks)}")
    for b in specificity_breaks:
        print(f"  ⚠  {b}")
    if not specificity_breaks:
        print("  ✓ None — confirmed-ineligible patients did not leak to ELIGIBLE")

asyncio.run(run())
