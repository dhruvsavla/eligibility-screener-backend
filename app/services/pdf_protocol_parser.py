"""
PDFProtocolParser — downloads and parses the full protocol PDF
for a given NCT ID from ClinicalTrials.gov, extracting richer
I/E criteria than the API summary provides.

Pipeline:
  1. Query CT.gov API v2 for the study's document URLs
  2. If a protocol PDF exists → download it
  3. Extract text using pdfplumber (primary) with PyMuPDF fallback
  4. locate_eligibility_section() finds inclusion+exclusion in FULL text
  5. Merge PDF-extracted criteria with API criteria (deduplicate)
  6. Return enriched criteria text

Fallback chain:
  PDF available and parseable → use PDF text
  PDF available but unreadable (scanned/locked) → use API text + log WARNING
  No PDF linked → use API text + log INFO (normal for many trials)
"""

import io
import re
import time
from loguru import logger
from app.config import settings


class PDFExtractionError(Exception):
    """Both pdfplumber and PyMuPDF failed to extract text."""


class PDFDownloadError(Exception):
    """HTTP error downloading the PDF."""


class PDFProtocolParser:

    def get_protocol_pdf_url(self, nct_id: str) -> str | None:
        """Query CT.gov API v2 for the study's protocol PDF URL."""
        import requests

        logger.info("Checking for protocol PDF for {}...", nct_id)
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields=hasResults,documents"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("⚠ Could not query document metadata for {}: {}", nct_id, e)
            return None

        data = resp.json()
        doc_section = data.get("documentSection", {})
        large_doc_module = doc_section.get("largeDocumentModule", {})
        large_docs = large_doc_module.get("largeDocs", [])

        for doc in large_docs:
            type_abbrev = doc.get("typeAbbrev", "")
            filename = doc.get("filename", "")
            if type_abbrev in ("Prot", "Prot_SAP") or filename.lower().endswith(".pdf"):
                size_kb = round(doc.get("size", 0) / 1024, 1)
                upload_date = doc.get("date", "")
                pdf_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}/documents/{filename}"
                logger.info(
                    "✓ Found protocol PDF: {} ({} KB, uploaded {})",
                    filename, size_kb, upload_date
                )
                return pdf_url

        logger.info("No protocol PDF linked for {} — using API text", nct_id)
        return None

    def download_and_extract_text(self, pdf_url: str) -> str:
        """Download PDF and extract text using pdfplumber then PyMuPDF fallback."""
        import requests

        logger.info("Downloading PDF from {}...", pdf_url[:80])
        start = time.time()
        try:
            resp = requests.get(pdf_url, timeout=30, stream=True)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise PDFDownloadError(f"HTTP error downloading PDF: {e}") from e
        except requests.exceptions.RequestException as e:
            raise PDFDownloadError(f"Network error downloading PDF: {e}") from e

        pdf_bytes = resp.content
        elapsed = int((time.time() - start) * 1000)
        logger.info("✓ Downloaded PDF in {}ms ({} KB)", elapsed, round(len(pdf_bytes) / 1024, 1))

        # Try pdfplumber first
        method = "pdfplumber"
        text = ""
        n_pages = 0
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                n_pages = len(pdf.pages)
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.warning("⚠ pdfplumber failed: {} — trying PyMuPDF", e)
            text = ""

        # PyMuPDF fallback if pdfplumber got < 100 chars (scanned PDF)
        if len(text.strip()) < 100:
            method = "PyMuPDF"
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                n_pages = len(doc)
                text = "\n".join(page.get_text() for page in doc)
            except Exception as e:
                logger.error("✗ PyMuPDF also failed: {}", e)
                raise PDFExtractionError(
                    f"Both pdfplumber and PyMuPDF failed to extract text from PDF"
                ) from e

        if len(text.strip()) < 100:
            raise PDFExtractionError("Extracted text is too short — likely a scanned/image PDF")

        logger.info(
            "Extracted {} chars from {}-page PDF using {}",
            len(text), n_pages, method
        )
        return text

    def locate_eligibility_section(self, full_text: str, max_section_chars: int = 40000) -> str:
        """
        Locate the inclusion+exclusion criteria section in FULL protocol text using
        pure-Python regex — no character limit, no Claude call.

        Key challenge: Table of Contents entries also contain "inclusion criteria" /
        "exclusion criteria" but are fake (just page references like "8.1 Inclusion
        Criteria ............ 30").  We detect and skip ToC entries by checking for
        5+ consecutive dots within 200 chars after the marker.

        Strategy:
          1. Find ALL "inclusion criteria" positions that are NOT ToC entries.
          2. For each real candidate, find the nearest following "exclusion criteria"
             that is also not a ToC entry, within 30 000 chars.
          3. Start at the section header above the inclusion marker (back up to the
             preceding newline, up to 300 chars).
          4. End after the exclusion block: first major section header found after
             the exclusion start, or +20 000 chars if none found.
          5. Cap the returned section at max_section_chars.
        """
        lower = full_text.lower()

        def _is_toc_entry(pos: int) -> bool:
            # ToC entries have trailing dots ("........30") within 200 chars after the marker.
            ahead = lower[pos: pos + 200]
            return bool(re.search(r"\.{5,}", ahead))

        def _is_section_header(pos: int) -> bool:
            # Real section headers have ONLY whitespace or a section number (e.g. "8.1.")
            # between the preceding newline and the marker.
            # Inline mentions like "Entered patients who meet the inclusion criteria"
            # have full words on the same line before the marker.
            before = lower[max(0, pos - 200): pos]
            last_nl = before.rfind("\n")
            if last_nl == -1:
                # No preceding newline: valid if text starts here (API-style criteria
                # that begins with "Inclusion Criteria:") or the before slice is whitespace only.
                return pos <= 5 or len(before.strip()) == 0
            same_line = before[last_nl + 1:].strip()
            # Allow empty line (header alone) or section numbers like "8.1.", "1.", "(a)"
            return bool(re.match(r'^[\d.()\s]*$', same_line))

        incl_positions = [m.start() for m in re.finditer(r"inclusion criteria", lower)
                          if not _is_toc_entry(m.start()) and _is_section_header(m.start())]
        excl_positions = [m.start() for m in re.finditer(r"exclusion criteria", lower)
                          if not _is_toc_entry(m.start())]

        logger.info(
            "locate_eligibility_section: {} real incl markers, {} real excl markers (ToC entries skipped)",
            len(incl_positions), len(excl_positions),
        )

        if not incl_positions and not excl_positions:
            logger.warning("No real eligibility markers found — returning full text (capped)")
            return full_text[:max_section_chars]

        # Find an inclusion marker paired with a following exclusion section header.
        # Minimum 500-char gap filters out inline mentions like "meet the inclusion criteria
        # and do not meet any of the exclusion criteria" which are only ~50 chars apart.
        MIN_SECTION_GAP = 500
        start = None
        for inc_pos in incl_positions:
            following_excl = [e for e in excl_positions
                              if inc_pos + MIN_SECTION_GAP < e < inc_pos + 30000]
            if following_excl:
                start = inc_pos
                logger.info(
                    "Paired eligibility section: inclusion at char {:,}, exclusion at char {:,}",
                    inc_pos, following_excl[0],
                )
                break

        if start is None:
            # Fallback: use last real inclusion marker (or first exclusion marker).
            if incl_positions:
                start = incl_positions[-1]
                logger.warning("No paired incl/excl — using last inclusion marker at {:,}", start)
            else:
                start = excl_positions[0]
                logger.warning("No inclusion marker — starting at first exclusion at {:,}", start)

        # Back up to the nearest newline above start to capture the section header.
        header_lookback = full_text.rfind("\n", max(0, start - 300), start)
        if header_lookback != -1:
            start = header_lookback

        # Find the END: next PEER or HIGHER section heading after the exclusion marker.
        # We deliberately search for peer/higher sections only (e.g. "8.3 Discontinuations"
        # after "8.2 Exclusion Criteria"), and NOT sub-sections like "8.2.1 Rationale"
        # which are WITHIN the exclusion section and should be included.
        real_excl_in_range = [e for e in excl_positions if e >= start]
        first_excl = real_excl_in_range[0] if real_excl_in_range else None

        end = len(full_text)
        if first_excl is not None:
            # Search starts well after the exclusion header to skip sub-section content.
            # Use a generous buffer (2,000 chars) before looking for the next section.
            search_from = first_excl + 2000
            # Patterns for PEER/HIGHER level sections only.
            # "discontinuations", "study procedures", "treatment", etc. as standalone headings.
            # Avoid matching "8.2.1" style sub-sections — use \b word-boundary patterns
            # for English section names rather than numeric patterns.
            end_patterns = [
                r"\ndiscontinuation",      # "8.3 Discontinuations" / "Discontinuation"
                r"\nstudy procedures",
                r"\nstudy visits",
                r"\nvisit schedule",
                r"\ntreatment period",
                r"\nrandomization\s*\n",
                r"\npharmacoki",
                r"\nstatistical",
                r"\nadverse event",
                r"\nappendix",
            ]
            candidate_ends = []
            sub = lower[search_from:]
            for pat in end_patterns:
                m = re.search(pat, sub)
                if m:
                    candidate_ends.append(search_from + m.start())
            if candidate_ends:
                end = min(candidate_ends)
                logger.info("Eligibility section ends at char {:,} (next section header)", end)
            else:
                # Generous safety cap after the exclusion start — eligibility sections
                # are typically ≤ 15 000 chars even in large protocols.
                end = min(first_excl + 15000, len(full_text))
                logger.info("No section-end marker found — using +15k cap at {:,}", end)

        section = full_text[start:end]
        if len(section) > max_section_chars:
            logger.warning(
                "Located section ({:,} chars) exceeds cap {:,} — truncating",
                len(section), max_section_chars,
            )
            section = section[:max_section_chars]

        logger.info(
            "✓ Located eligibility section: {:,} chars (chars {:,}–{:,} of {:,} total)",
            len(section), start, end, len(full_text),
        )
        return section

    def extract_eligibility_section(self, full_pdf_text: str, nct_id: str) -> str:
        """
        Locate and return the eligibility criteria section from full PDF text.
        Uses pure-Python locate_eligibility_section() on the FULL text — no truncation
        before the search, so the real section is found even in 200-page documents.
        """
        logger.info("Locating eligibility section in PDF for {} ({:,} chars)...",
                    nct_id, len(full_pdf_text))
        section = self.locate_eligibility_section(full_pdf_text)
        logger.info("Located eligibility section: {:,} chars", len(section))
        return section

    def merge_criteria(
        self, api_criteria_text: str, pdf_criteria_text: str, nct_id: str
    ) -> str:
        """Use Claude Sonnet to merge API and PDF criteria, preferring more specific versions."""
        from app.services.llm_client import get_claude_client

        api_len = len(api_criteria_text)
        pdf_len = len(pdf_criteria_text)
        logger.info("Merging API ({} chars) + PDF ({} chars) criteria...", api_len, pdf_len)

        system = (
            "You are a clinical trial expert. You have two versions of eligibility "
            "criteria for the same trial: one from the API summary (shorter) and one from "
            "the full protocol PDF (more detailed). Merge them into one complete list.\n"
            "- Keep all criteria from both sources\n"
            "- Where they overlap, keep the MORE SPECIFIC version (e.g. 'eGFR >= 45 mL/min' "
            "is more specific than 'no renal impairment')\n"
            "- Preserve the Inclusion/Exclusion structure\n"
            "- Return plain text, no markdown"
        )
        user = (
            f"API summary criteria for {nct_id}:\n{api_criteria_text}\n\n"
            f"---\nFull protocol PDF criteria:\n{pdf_criteria_text}"
        )
        try:
            merged = get_claude_client().complete(system, user) or api_criteria_text
            merged_len = len(merged)
            delta = merged_len - api_len
            logger.info(
                "Merged API ({} chars) + PDF ({} chars) → {} chars",
                api_len, pdf_len, merged_len
            )
            logger.info(
                "PDF enrichment added approximately {} additional chars of criteria detail", delta
            )
            return merged
        except Exception as e:
            logger.warning("⚠ Claude Sonnet merge failed: {} — returning API text", e)
            return api_criteria_text

    def parse_protocol(self, nct_id: str, api_criteria_text: str) -> str:
        """
        Full pipeline: attempt PDF download + extraction, merge with API text.
        Returns the best available criteria text.
        """
        logger.info("INFO  | Fetching protocol {} from ClinicalTrials.gov API...", nct_id)
        logger.info("INFO  | ✓ API returned {} chars of eligibility criteria", len(api_criteria_text))

        pdf_url = self.get_protocol_pdf_url(nct_id)
        if not pdf_url:
            logger.info("INFO  | ✓ Protocol ingestion complete for {} (API text only)", nct_id)
            return api_criteria_text

        logger.info("INFO  | Downloading PDF...")
        try:
            full_text = self.download_and_extract_text(pdf_url)
        except PDFDownloadError as e:
            logger.warning("⚠ PDF download failed: {} — using API text", e)
            return api_criteria_text
        except PDFExtractionError as e:
            logger.warning("⚠ PDF extraction failed: {} — using API text", e)
            return api_criteria_text

        pdf_eligibility = self.extract_eligibility_section(full_text, nct_id)
        if not pdf_eligibility.strip():
            logger.warning("⚠ No eligibility section found in PDF — using API text")
            return api_criteria_text

        merged = self.merge_criteria(api_criteria_text, pdf_eligibility, nct_id)
        logger.info("INFO  | ✓ Protocol ingestion complete for {}", nct_id)
        return merged
