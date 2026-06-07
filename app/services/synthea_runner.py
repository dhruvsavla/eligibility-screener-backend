"""
SyntheaRunner — wraps the official MITRE Synthea Java tool.

Synthea is a Java application that must be downloaded separately.
This runner:
  1. Checks if Java is available (java --version)
  2. Checks if synthea.jar exists in backend/synthea/synthea.jar
  3. If both present: runs Synthea to generate real FHIR R4 bundles
  4. If not: clearly logs WHY it cannot run and raises SyntheaNotAvailableError
     so the caller can fall back to synthea_generator.py

The caller (patients router) handles the fallback transparently.
"""

import glob
import json
import os
import re
import subprocess
import time
from loguru import logger

SYNTHEA_JAR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "synthea", "synthea.jar"
)
SYNTHEA_JAR_PATH = os.path.normpath(SYNTHEA_JAR_PATH)


class SyntheaNotAvailableError(Exception):
    """Java or synthea.jar is missing — cannot run real Synthea."""


class SyntheaGenerationError(Exception):
    """Synthea exited with non-zero code."""


class SyntheaRunner:

    def check_prerequisites(self) -> dict:
        """Return a full prerequisites report for Synthea."""
        logger.info("=== SYNTHEA PREREQUISITES CHECK ===")

        # Check Java
        java_available = False
        java_version = None
        try:
            result = subprocess.run(
                ["java", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            java_available = result.returncode == 0
            version_match = re.search(r"(\d+\.\d+[\.\d]*)", output)
            if version_match:
                java_version = version_match.group(1)
        except FileNotFoundError:
            java_available = False
        except subprocess.TimeoutExpired:
            java_available = False

        # Check synthea JAR
        synthea_jar_exists = os.path.isfile(SYNTHEA_JAR_PATH)
        synthea_jar_size_mb = None
        if synthea_jar_exists:
            synthea_jar_size_mb = round(os.path.getsize(SYNTHEA_JAR_PATH) / (1024 * 1024), 1)

        can_run = java_available and synthea_jar_exists

        if java_available:
            logger.info("Java available: ✓ (version {})", java_version or "unknown")
        else:
            logger.warning("Java available: ✗ — java not found in PATH")

        if synthea_jar_exists:
            logger.info(
                "Synthea JAR: ✓ found at {} ({} MB)", SYNTHEA_JAR_PATH, synthea_jar_size_mb
            )
        else:
            logger.warning("Synthea JAR: ✗ — not found at {}", SYNTHEA_JAR_PATH)

        if can_run:
            reason = f"Java {java_version} found and synthea.jar ({synthea_jar_size_mb} MB) present"
            logger.info("Status: READY — real Synthea generation available")
        else:
            parts = []
            if not java_available:
                parts.append("java not found in PATH")
            if not synthea_jar_exists:
                parts.append(f"synthea.jar not found at {SYNTHEA_JAR_PATH}")
            reason = "; ".join(parts)
            logger.warning("Status: NOT AVAILABLE — will use Python fallback generator")
            logger.warning("  Reason: {}", reason)
            logger.warning("  To enable real Synthea: see backend/synthea/SETUP.md")

        return {
            "java_available": java_available,
            "java_version": java_version,
            "java_min_version": 11,
            "synthea_jar_exists": synthea_jar_exists,
            "synthea_jar_path": SYNTHEA_JAR_PATH,
            "synthea_jar_size_mb": synthea_jar_size_mb,
            "can_run": can_run,
            "reason": reason,
        }

    def generate(
        self,
        count: int,
        seed: int = 42,
        output_dir: str | None = None,
    ) -> list[dict]:
        """Run Synthea and return parsed FHIR R4 bundle dicts."""
        prereqs = self.check_prerequisites()
        if not prereqs["can_run"]:
            raise SyntheaNotAvailableError(prereqs["reason"])

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(SYNTHEA_JAR_PATH), "output"
            )

        os.makedirs(output_dir, exist_ok=True)
        fhir_dir = os.path.join(output_dir, "fhir")

        cmd = [
            "java",
            "-jar",
            SYNTHEA_JAR_PATH,
            "-p",
            str(count),
            "-s",
            str(seed),
            "--exporter.fhir.export=true",
            "--exporter.fhir.transaction_bundle=false",
            f"--exporter.baseDirectory={output_dir}",
            "Massachusetts",
        ]

        logger.info("Running Synthea: count={} seed={} output={}", count, seed, output_dir)
        logger.debug("Command: {}", " ".join(cmd))
        start = time.time()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            raise SyntheaGenerationError("Synthea timed out after 300s") from e

        for line in proc.stdout.splitlines():
            logger.info("[SYNTHEA] {}", line)
        for line in proc.stderr.splitlines():
            logger.warning("[SYNTHEA] {}", line)

        elapsed = int((time.time() - start) * 1000)

        if proc.returncode != 0:
            raise SyntheaGenerationError(
                f"Synthea exited with code {proc.returncode}. stderr: {proc.stderr[:500]}"
            )

        # Load generated FHIR bundles
        pattern = os.path.join(fhir_dir, "*.json")
        fhir_files = glob.glob(pattern)

        bundles = []
        for fpath in fhir_files:
            try:
                with open(fpath) as f:
                    bundle = json.load(f)
                if bundle.get("resourceType") == "Bundle":
                    bundles.append(bundle)
            except Exception as e:
                logger.warning("⚠ Could not load Synthea output {}: {}", fpath, e)

        logger.info(
            "[SYNTHEA] Generation complete: {} patient files written to {} ({}ms)",
            len(bundles), output_dir, elapsed,
        )
        return bundles

    def get_setup_instructions(self) -> str:
        setup_path = os.path.join(os.path.dirname(SYNTHEA_JAR_PATH), "SETUP.md")
        if os.path.isfile(setup_path):
            with open(setup_path) as f:
                return f.read()
        return (
            "See backend/synthea/SETUP.md for instructions on downloading and configuring Synthea."
        )


synthea_runner = SyntheaRunner()
