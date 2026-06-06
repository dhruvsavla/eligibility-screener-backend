import time
import requests
from loguru import logger
from fastapi import HTTPException
from app.config import settings


class ClinicalTrialsClient:
    def __init__(self):
        self.base_url = settings.CLINICALTRIALS_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch_trials(self, condition: str, phase: str, count: int) -> list[dict]:
        url = f"{self.base_url}/studies"
        # API v2 uses query.term with AREA syntax for phase filtering
        params = {
            "query.cond": condition,
            "query.term": f"AREA[Phase]{phase}",
            "pageSize": count,
            "format": "json",
        }
        logger.info(
            "Fetching trials for condition='{}' phase='{}' count={}", condition, phase, count
        )
        logger.debug("GET {} phase_filter=AREA[Phase]{} params={}", url, phase, params)

        start = time.time()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            elapsed = int((time.time() - start) * 1000)
            logger.info("ClinicalTrials.gov → HTTP {} in {}ms", resp.status_code, elapsed)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("ClinicalTrials.gov HTTP error: {}", e)
            raise HTTPException(status_code=502, detail=f"ClinicalTrials.gov API error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error("ClinicalTrials.gov request failed: {}", e)
            time.sleep(2)
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as retry_err:
                raise HTTPException(
                    status_code=502, detail=f"ClinicalTrials.gov unreachable: {retry_err}"
                )

        data = resp.json()
        studies = data.get("studies", [])
        logger.info("✓ Fetched {} trials in {}ms", len(studies), elapsed)

        results = []
        for study in studies:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            cond_mod = proto.get("conditionsModule", {})
            elig_mod = proto.get("eligibilityModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

            nct_id = id_mod.get("nctId", "")
            title = id_mod.get("briefTitle", "")
            conditions = cond_mod.get("conditions", [])
            criteria_text = elig_mod.get("eligibilityCriteria", "")
            phases = design_mod.get("phases", [])
            lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")

            logger.info(
                "  Trial {}: '{}' | criteria preview: {}...",
                nct_id,
                title[:60],
                criteria_text[:200],
            )

            results.append(
                {
                    "nct_id": nct_id,
                    "title": title,
                    "condition": ", ".join(conditions),
                    "phase": ", ".join(phases) if phases else phase,
                    "sponsor": lead_sponsor,
                    "eligibility_criteria": criteria_text,
                }
            )

        return results

    def fetch_by_nct_id(self, nct_id: str) -> dict:
        url = f"{self.base_url}/studies/{nct_id}"
        logger.info("Fetching trial by NCT ID: {}", nct_id)
        start = time.time()
        try:
            resp = self.session.get(url, timeout=30)
            elapsed = int((time.time() - start) * 1000)
            logger.info("ClinicalTrials.gov → HTTP {} in {}ms", resp.status_code, elapsed)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch NCT {}: {}", nct_id, e)
            raise HTTPException(status_code=502, detail=f"Cannot fetch {nct_id}: {e}")

        data = resp.json()
        proto = data.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        cond_mod = proto.get("conditionsModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        design_mod = proto.get("designModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        criteria_text = elig_mod.get("eligibilityCriteria", "")
        logger.info(
            "✓ Fetched {}: criteria preview: {}...", nct_id, criteria_text[:200]
        )

        return {
            "nct_id": id_mod.get("nctId", nct_id),
            "title": id_mod.get("briefTitle", ""),
            "condition": ", ".join(cond_mod.get("conditions", [])),
            "phase": ", ".join(design_mod.get("phases", [])),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "eligibility_criteria": criteria_text,
        }


clinical_trials_client = ClinicalTrialsClient()
