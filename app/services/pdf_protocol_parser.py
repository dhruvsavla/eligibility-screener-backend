"""
PDFProtocolParser — downloads and parses the full protocol PDF
for a given NCT ID from ClinicalTrials.gov, extracting richer
I/E criteria than the API summary provides.

Pipeline:
  1. Query CT.gov API v2 for the study's document URLs
  2. If a protocol PDF exists → download it
  3. Extract text using pdfplumber (primary) with PyMuPDF fallback
  4. Use GPT-4o to locate and extract the Eligibility section
  5. Merge PDF-extracted criteria with API criteria (deduplicate)
  6. Return enriched criteria text

Fallback chain:
  PDF available and parseable → use PDF text
  PDF available but unreadable (scanned/locked) → use API text + log WARNING
  No PDF linked → use API text + log INFO (normal for many trials)
"""

import io
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

    def extract_eligibility_section(self, full_pdf_text: str, nct_id: str) -> str:
        """Use GPT-4o to locate and return only the eligibility criteria section."""
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("Using GPT-4o to locate eligibility section in PDF for {}...", nct_id)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical document parser. Given full protocol PDF text, "
                            "extract ONLY the section containing Inclusion Criteria and Exclusion Criteria. "
                            "Return the raw text of that section verbatim. Do not summarize or interpret. "
                            "If no eligibility section is found, return an empty string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extract the eligibility criteria section from this protocol PDF for {nct_id}:\n\n"
                            f"{full_pdf_text[:15000]}"
                        ),
                    },
                ],
            )
            section = response.choices[0].message.content or ""
            logger.info("GPT-4o located eligibility section: {} chars", len(section))
            return section
        except Exception as e:
            logger.warning("⚠ GPT-4o failed to extract eligibility section: {}", e)
            return ""

    def merge_criteria(
        self, api_criteria_text: str, pdf_criteria_text: str, nct_id: str
    ) -> str:
        """Use GPT-4o to merge API and PDF criteria, preferring more specific versions."""
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        api_len = len(api_criteria_text)
        pdf_len = len(pdf_criteria_text)
        logger.info("Merging API ({} chars) + PDF ({} chars) criteria...", api_len, pdf_len)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical trial expert. You have two versions of eligibility "
                            "criteria for the same trial: one from the API summary (shorter) and one from "
                            "the full protocol PDF (more detailed). Merge them into one complete list.\n"
                            "- Keep all criteria from both sources\n"
                            "- Where they overlap, keep the MORE SPECIFIC version (e.g. 'eGFR >= 45 mL/min' "
                            "is more specific than 'no renal impairment')\n"
                            "- Preserve the Inclusion/Exclusion structure\n"
                            "- Return plain text, no markdown"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"API summary criteria for {nct_id}:\n{api_criteria_text}\n\n"
                            f"---\nFull protocol PDF criteria:\n{pdf_criteria_text}"
                        ),
                    },
                ],
            )
            merged = response.choices[0].message.content or api_criteria_text
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
            logger.warning("⚠ GPT-4o merge failed: {} — returning API text", e)
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
