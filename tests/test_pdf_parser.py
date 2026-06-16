"""Tests for pdf_protocol_parser.py"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.pdf_protocol_parser import (
    PDFProtocolParser,
    PDFDownloadError,
    PDFExtractionError,
)


@pytest.fixture
def parser():
    return PDFProtocolParser()


# Test 1: get_protocol_pdf_url — API response containing a PDF doc
def test_get_protocol_pdf_url_found(parser):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "documentSection": {
            "largeDocumentModule": {
                "largeDocs": [
                    {
                        "typeAbbrev": "Prot",
                        "filename": "prot_000.pdf",
                        "size": 2_400_000,
                        "date": "2023-01-15",
                    }
                ]
            }
        }
    }

    with patch("requests.get", return_value=mock_response):
        url = parser.get_protocol_pdf_url("NCT04280783")

    assert url is not None
    assert "prot_000.pdf" in url


# Test 2: get_protocol_pdf_url — no PDF → returns None
def test_get_protocol_pdf_url_not_found(parser):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "documentSection": {"largeDocumentModule": {"largeDocs": []}}
    }

    with patch("requests.get", return_value=mock_response):
        url = parser.get_protocol_pdf_url("NCT00000000")

    assert url is None


# Test 3: download_and_extract_text with pdfplumber succeeding
def test_download_and_extract_text_pdfplumber(parser):
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = fake_pdf_bytes

    mock_page = MagicMock()
    # Must exceed the 100-char threshold or the parser falls through to PyMuPDF.
    mock_page.extract_text.return_value = (
        "Inclusion Criteria:\n- Age 18-75 years\n- HbA1c >= 7.5% at screening\n"
        "- Type 2 diabetes mellitus diagnosis\nExclusion Criteria:\n- Current insulin use\n- Pregnancy"
    )
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("requests.get", return_value=mock_resp):
        with patch("pdfplumber.open", return_value=mock_pdf):
            text = parser.download_and_extract_text("https://example.com/prot.pdf")

    assert "Inclusion Criteria" in text
    assert "HbA1c" in text


# Test 4: download_and_extract_text with empty bytes → PDFExtractionError
def test_download_and_extract_text_empty_raises(parser):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b""

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    mock_fitz_doc = MagicMock()
    mock_fitz_doc.__len__ = lambda self: 0
    mock_fitz_doc.__iter__ = lambda self: iter([])

    with patch("requests.get", return_value=mock_resp):
        with patch("pdfplumber.open", return_value=mock_pdf):
            with patch("fitz.open", return_value=mock_fitz_doc):
                with pytest.raises(PDFExtractionError):
                    parser.download_and_extract_text("https://example.com/empty.pdf")


# Test 5: merge_criteria combines two texts via mocked Claude Sonnet
def test_merge_criteria_mocked_claude(parser):
    api_text = "Inclusion:\n- Age 18-75\nExclusion:\n- Pregnancy"
    pdf_text = "Inclusion:\n- Age 18-75\n- HbA1c >= 7.5%\nExclusion:\n- Pregnancy\n- Active cancer"

    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "Inclusion:\n- Age 18-75\n- HbA1c >= 7.5%\nExclusion:\n- Pregnancy\n- Active cancer"
    )

    with patch("app.services.llm_client.get_claude_client", return_value=mock_client):
        merged = parser.merge_criteria(api_text, pdf_text, "NCT00000001")

    assert "HbA1c" in merged
    assert "Active cancer" in merged
    assert len(merged) > len(api_text)


# Test 6: extract_eligibility_section via mocked Claude Sonnet
def test_extract_eligibility_section_mocked_claude(parser):
    full_text = "Title page...\nInclusion Criteria:\n- Age 18-75\nExclusion Criteria:\n- Pregnancy"
    mock_client = MagicMock()
    mock_client.complete.return_value = "Inclusion Criteria:\n- Age 18-75\nExclusion Criteria:\n- Pregnancy"

    with patch("app.services.llm_client.get_claude_client", return_value=mock_client):
        section = parser.extract_eligibility_section(full_text, "NCT00000002")

    assert "Inclusion Criteria" in section
