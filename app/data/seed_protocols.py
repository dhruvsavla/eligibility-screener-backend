"""
Seeds 10 real ClinicalTrials.gov protocols across diabetes, oncology, cardiology.
Protocol-agnostic: edit PROTOCOL_NCT_IDS to use any protocols.
The FLAGSHIP (first diabetes entry) carries the 100-patient ground truth set.

Run: python -m app.data.seed_protocols
"""
import asyncio
from loguru import logger

# Edit this list to seed any protocols. The system works for ALL of them.
PROTOCOL_NCT_IDS = {
    "diabetes": [
        "NCT01394952",  # FLAGSHIP — set programmatically below; replace with any T2DM trial
        "NCT00529815",
        "NCT01147627",
    ],
    "oncology": [
        "NCT02499770",
        "NCT02395172",
        "NCT01871454",
    ],
    "cardiology": [
        "NCT01032629",
        "NCT00831441",
        "NCT01206972",
        "NCT00457392",
    ],
}

FLAGSHIP_AREA = "diabetes"   # first entry in this area carries ground truth


async def seed_all() -> dict:
    """Fetch each NCT ID, run it through the LangChain agent, and persist it.

    Bad / 404 / empty NCT IDs are logged and skipped — never abort the run.
    The first successfully-seeded protocol in FLAGSHIP_AREA is marked is_flagship=1.
    """
    from app import database as db
    from app.services.trials_fetcher import clinical_trials_client
    from app.routers.protocols import _ingest_criteria, _save_protocol_and_rules

    seeded: list[dict] = []
    failures: list[dict] = []
    flagship_id: int | None = None

    # Clear any previous flagship flag so exactly one ends up set.
    try:
        await db.execute("UPDATE protocols SET is_flagship = 0")
    except Exception:
        pass

    # Iterate FLAGSHIP_AREA first so its first success becomes the flagship.
    areas = [FLAGSHIP_AREA] + [a for a in PROTOCOL_NCT_IDS if a != FLAGSHIP_AREA]

    total = sum(len(v) for v in PROTOCOL_NCT_IDS.values())
    logger.info("=== SEEDING {} PROTOCOLS ({}) ===", total, ", ".join(areas))

    for area in areas:
        for nct_id in PROTOCOL_NCT_IDS.get(area, []):
            try:
                trial = clinical_trials_client.fetch_by_nct_id(nct_id)
            except Exception as e:
                logger.warning("Skipping {} ({}): fetch failed — {}", nct_id, area, e)
                failures.append({"nct_id": nct_id, "area": area, "reason": f"fetch failed: {e}"})
                continue

            raw_text = trial.get("eligibility_criteria", "")
            if not raw_text or not raw_text.strip():
                logger.warning("Skipping {} ({}): empty eligibility criteria", nct_id, area)
                failures.append({"nct_id": nct_id, "area": area, "reason": "empty criteria"})
                continue

            try:
                rules, agent_trace = _ingest_criteria(raw_text, nct_id)
                proto = await _save_protocol_and_rules(
                    nct_id=nct_id,
                    title=trial.get("title", ""),
                    condition=trial.get("condition", area),
                    phase=trial.get("phase", ""),
                    sponsor=trial.get("sponsor", ""),
                    raw_criteria_text=raw_text,
                    rules=rules,
                    agent_trace=agent_trace,
                )
            except Exception as e:
                logger.warning("Skipping {} ({}): ingestion failed — {}", nct_id, area, e)
                failures.append({"nct_id": nct_id, "area": area, "reason": f"ingestion failed: {e}"})
                continue

            pid = proto.get("id")
            is_flag = False
            if area == FLAGSHIP_AREA and flagship_id is None and pid:
                flagship_id = pid
                is_flag = True
                await db.execute("UPDATE protocols SET is_flagship = 1 WHERE id = ?", (pid,))
                logger.info("★ FLAGSHIP set: {} (id={}) — {}", nct_id, pid, trial.get("title", "")[:60])

            seeded.append({
                "id": pid, "nct_id": nct_id, "area": area,
                "title": trial.get("title", ""), "rules": len(rules),
                "is_flagship": is_flag,
            })
            logger.info("✓ Seeded {} ({}) — {} rules", nct_id, area, len(rules))

    logger.info(
        "=== SEED COMPLETE: {}/{} seeded, {} failed | flagship_id={} ===",
        len(seeded), total, len(failures), flagship_id,
    )
    if failures:
        for f in failures:
            logger.warning("  failed: {} ({}) — {}", f["nct_id"], f["area"], f["reason"])

    return {
        "seeded": seeded,
        "failed": failures,
        "seeded_count": len(seeded),
        "failed_count": len(failures),
        "flagship_id": flagship_id,
    }


async def _main():
    from app.database import init_db
    from app.routers.evaluation import _ensure_gt_columns
    await init_db()
    await _ensure_gt_columns()
    result = await seed_all()
    print(f"\nSeeded {result['seeded_count']} protocols, "
          f"{result['failed_count']} failed. Flagship id = {result['flagship_id']}")


if __name__ == "__main__":
    asyncio.run(_main())
