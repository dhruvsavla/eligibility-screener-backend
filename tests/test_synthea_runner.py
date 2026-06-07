"""Tests for synthea_runner.py"""

import json
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from app.services.synthea_runner import (
    SyntheaRunner,
    SyntheaNotAvailableError,
    SyntheaGenerationError,
    SYNTHEA_JAR_PATH,
)


@pytest.fixture
def runner():
    return SyntheaRunner()


# Test 1: check_prerequisites when java not in PATH
def test_check_prerequisites_no_java(runner):
    with patch("subprocess.run", side_effect=FileNotFoundError("java not found")):
        with patch("os.path.isfile", return_value=False):
            prereqs = runner.check_prerequisites()

    assert prereqs["java_available"] is False
    assert prereqs["synthea_jar_exists"] is False
    assert prereqs["can_run"] is False
    assert "java" in prereqs["reason"].lower() or "not found" in prereqs["reason"].lower()


# Test 2: check_prerequisites when JAR missing but Java present
def test_check_prerequisites_no_jar(runner):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "java version 17.0.8"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        with patch("os.path.isfile", return_value=False):
            prereqs = runner.check_prerequisites()

    assert prereqs["java_available"] is True
    assert prereqs["synthea_jar_exists"] is False
    assert prereqs["can_run"] is False


# Test 3: check_prerequisites with mocked java and JAR
def test_check_prerequisites_all_present(runner):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "openjdk 17.0.8 2023-07-18"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        with patch("os.path.isfile", return_value=True):
            with patch("os.path.getsize", return_value=150 * 1024 * 1024):
                prereqs = runner.check_prerequisites()

    assert prereqs["java_available"] is True
    assert prereqs["synthea_jar_exists"] is True
    assert prereqs["can_run"] is True


# Test 4: generate with mocked subprocess that returns 0 → reads output files
def test_generate_success(runner):
    # Create temp dir with a fake FHIR file
    with tempfile.TemporaryDirectory() as tmp:
        fhir_dir = os.path.join(tmp, "fhir")
        os.makedirs(fhir_dir)
        fake_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": "Patient", "id": "pt-1"}}],
        }
        with open(os.path.join(fhir_dir, "patient1.json"), "w") as f:
            json.dump(fake_bundle, f)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Generating 1 patient...\nDone."
        mock_proc.stderr = ""

        java_result = MagicMock()
        java_result.returncode = 0
        java_result.stdout = "openjdk 17.0.8"
        java_result.stderr = ""

        with patch("subprocess.run", side_effect=[java_result, mock_proc]):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.getsize", return_value=150 * 1024 * 1024):
                    bundles = runner.generate(count=1, seed=42, output_dir=tmp)

    assert len(bundles) == 1
    assert bundles[0]["resourceType"] == "Bundle"


# Test 5: generate with mocked subprocess returning non-zero → SyntheaGenerationError
def test_generate_nonzero_exit(runner):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "Error: OutOfMemoryError"

    java_result = MagicMock()
    java_result.returncode = 0
    java_result.stdout = "openjdk 17.0.8"
    java_result.stderr = ""

    with patch("subprocess.run", side_effect=[java_result, mock_proc]):
        with patch("os.path.isfile", return_value=True):
            with patch("os.path.getsize", return_value=150 * 1024 * 1024):
                with pytest.raises(SyntheaGenerationError):
                    runner.generate(count=5, seed=42)
