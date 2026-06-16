"""
AccuracyReportGenerator — generates a formatted accuracy report
as both JSON (for the API) and HTML (for download/display).
"""

from datetime import datetime
from loguru import logger
from app.services.evaluator import AccuracyMetrics


class AccuracyReportGenerator:

    def generate_html_report(
        self,
        metrics: AccuracyMetrics,
        protocol_title: str,
        eval_date: str | None = None,
    ) -> str:
        if eval_date is None:
            eval_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        sens_pct = round(metrics.sensitivity * 100, 1)
        spec_pct = round(metrics.specificity * 100, 1)
        ppv_pct  = round(metrics.ppv * 100, 1)
        npv_pct  = round(metrics.npv * 100, 1)
        f1_pct   = round(metrics.f1_score * 100, 1)
        acc_pct  = round(metrics.accuracy * 100, 1)

        sens_color = "#22c55e" if metrics.meets_sensitivity_target else "#ef4444"
        sens_badge = "TARGET MET ✓" if metrics.meets_sensitivity_target else "TARGET MISSED ✗"
        sens_badge_color = "#22c55e" if metrics.meets_sensitivity_target else "#ef4444"

        # Failure mode rows
        fm_rows = ""
        for mode, stats in metrics.failure_mode_accuracy.items():
            acc_val    = stats["accuracy"] if isinstance(stats, dict) else stats
            acc_pct_fm = round(acc_val * 100, 1)
            bar_color  = "#22c55e" if acc_val >= 0.85 else ("#f59e0b" if acc_val >= 0.70 else "#ef4444")
            mode_label = mode.replace("_", " ").title()
            fm_rows += f"""
            <tr>
              <td style="padding:8px 12px;color:#e2e8f0">{mode_label}</td>
              <td style="padding:8px 12px;text-align:right;color:{bar_color};font-weight:bold">{acc_pct_fm}%</td>
            </tr>"""

        # Confidence band
        cov_pct = round(metrics.confidence_coverage * 100, 1)
        br_pct  = round(metrics.borderline_review_rate * 100, 1)

        fp_total = metrics.false_positives_hard + metrics.false_positives_soft

        # Clinical interpretation
        interpretation = (
            f"The screener correctly identified {metrics.true_positives} of "
            f"{metrics.eligible_count} truly eligible patients "
            f"(sensitivity {sens_pct}%, {'exceeding' if metrics.meets_sensitivity_target else 'below'} "
            f"the {round(metrics.target_sensitivity * 100)}% target). "
            f"Note: under the clinical workflow definition used here, patients flagged as "
            f"REVIEW NEEDED are counted as correctly identified — a site coordinator reviews "
            f"these cases and enrolls truly eligible patients. Only patients incorrectly "
            f"predicted as INELIGIBLE are counted as misses (false negatives). "
            f"The screener produced {metrics.false_negatives} false negatives — truly eligible "
            f"patients predicted INELIGIBLE. These represent the highest-risk errors as they "
            f"would cause eligible patients to be excluded without coordinator review. "
            f"{metrics.false_positives_hard} hard false positives (predicted ELIGIBLE, truly not) "
            f"and {metrics.false_positives_soft} soft false positives (predicted REVIEW NEEDED, truly not — "
            f"coordinator review catches these) were also detected."
        )

        # Recommendations
        recs_html = "<ul style='color:#94a3b8;margin:0;padding-left:20px'>"
        if not metrics.meets_sensitivity_target:
            recs_html += "<li>Sensitivity below target — review HbA1c and eGFR lab alias matching</li>"
        if metrics.false_positives_hard > 0:
            recs_html += f"<li>{metrics.false_positives_hard} hard false positives detected — verify exclusion criterion detection (insulin, malignancy)</li>"
        if metrics.false_positives_soft > 0:
            recs_html += f"<li>{metrics.false_positives_soft} soft false positives (REVIEW_NEEDED for truly ineligible) — acceptable; coordinator will catch these</li>"
        low_mode_accs = [
            (m, s["accuracy"] if isinstance(s, dict) else s)
            for m, s in metrics.failure_mode_accuracy.items()
            if (s["accuracy"] if isinstance(s, dict) else s) < 0.80
        ]
        for mode, acc in low_mode_accs:
            recs_html += f"<li>Failure mode '{mode.replace('_',' ')}' accuracy {round(acc*100,1)}% — investigate concept matching for this criterion</li>"
        if metrics.borderline_review_rate < 0.70:
            recs_html += "<li>Low REVIEW_NEEDED rate for borderline patients — consider widening ambiguity detection</li>"
        if not low_mode_accs and metrics.meets_sensitivity_target:
            recs_html += "<li>Performance within targets — continue monitoring with real patient data</li>"
        recs_html += "</ul>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eligibility Screening Accuracy Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 32px; }}
  h2 {{ font-size: 14px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 16px; }}
  .card {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .metric-card {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px; padding: 20px; text-align: center; }}
  .metric-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px; }}
  .metric-value {{ font-size: 32px; font-weight: 800; }}
  .metric-sub {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; text-align: left; padding: 8px 12px; border-bottom: 1px solid #2a2d3a; }}
  td {{ font-size: 13px; border-bottom: 1px solid #1a1d27; }}
  .cm-cell {{ padding: 16px; text-align: center; border-radius: 8px; }}
  .cm-tp {{ background: #22c55e20; color: #22c55e; }}
  .cm-fn {{ background: #ef444420; color: #ef4444; }}
  .cm-fp {{ background: #f59e0b20; color: #f59e0b; }}
  .cm-tn {{ background: #22c55e20; color: #22c55e; }}
  .cm-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; color: inherit; opacity: .7; }}
  .cm-count {{ font-size: 28px; font-weight: 800; }}
  .cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 8px; }}
  .cm-header {{ display: flex; align-items: center; justify-content: center; padding: 8px; font-size: 11px; color: #64748b; font-weight: 600; }}
  p.interp {{ line-height: 1.7; color: #94a3b8; font-size: 13px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 11px; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
  <h1>Eligibility Screening Accuracy Report</h1>
  <p class="subtitle">
    Protocol: <strong style="color:#fff">{protocol_title}</strong> &nbsp;·&nbsp;
    Evaluated: <strong style="color:#fff">{eval_date}</strong> &nbsp;·&nbsp;
    Patients: <strong style="color:#fff">{metrics.total_evaluated}</strong>
  </p>

  <div class="metrics-grid">
    <div class="metric-card" style="border-color:{sens_color}40">
      <div class="metric-label">Sensitivity</div>
      <div class="metric-value" style="color:{sens_color}">{sens_pct}%</div>
      <div class="metric-sub">Target: {round(metrics.target_sensitivity*100)}%</div>
      <div style="margin-top:8px">
        <span class="badge" style="background:{sens_badge_color}20;color:{sens_badge_color}">{sens_badge}</span>
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Specificity</div>
      <div class="metric-value" style="color:#3b82f6">{spec_pct}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">F1 Score</div>
      <div class="metric-value" style="color:#a855f7">{f1_pct}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Accuracy</div>
      <div class="metric-value" style="color:#f59e0b">{acc_pct}%</div>
    </div>
  </div>

  <div class="card">
    <h2>Confusion Matrix</h2>
    <div class="cm-grid">
      <div></div>
      <div class="cm-header">Predicted ELIGIBLE</div>
      <div class="cm-header">Predicted NOT ELIGIBLE</div>
      <div class="cm-header" style="writing-mode:initial">True ELIGIBLE</div>
      <div class="cm-cell cm-tp">
        <div class="cm-count">{metrics.true_positives}</div>
        <div class="cm-label">True Positive</div>
      </div>
      <div class="cm-cell cm-fn">
        <div class="cm-count">{metrics.false_negatives}</div>
        <div class="cm-label">False Negative ← missed</div>
      </div>
      <div class="cm-header" style="writing-mode:initial">True NOT ELIGIBLE</div>
      <div class="cm-cell cm-fp">
        <div class="cm-count">{metrics.false_positives_hard}</div>
        <div class="cm-label">Hard FP ← coordinator misses</div>
        <div style="font-size:10px;margin-top:4px;opacity:.7">(+{metrics.false_positives_soft} soft FP → REVIEW)</div>
      </div>
      <div class="cm-cell cm-tn">
        <div class="cm-count">{metrics.true_negatives}</div>
        <div class="cm-label">True Negative</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Failure Mode Breakdown</h2>
    <table>
      <thead><tr><th>Failure Mode</th><th style="text-align:right">Accuracy</th></tr></thead>
      <tbody>{fm_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Confidence Band Calibration</h2>
    <table>
      <thead><tr><th>Metric</th><th style="text-align:right">Value</th></tr></thead>
      <tbody>
        <tr><td style="padding:8px 12px;color:#94a3b8">Mean confidence band width</td>
            <td style="padding:8px 12px;text-align:right;color:#e2e8f0">{metrics.mean_confidence_width:.1f} pts</td></tr>
        <tr><td style="padding:8px 12px;color:#94a3b8">Borderline → REVIEW_NEEDED rate</td>
            <td style="padding:8px 12px;text-align:right;color:#e2e8f0">{br_pct}%</td></tr>
        <tr><td style="padding:8px 12px;color:#94a3b8">Confidence coverage</td>
            <td style="padding:8px 12px;text-align:right;color:#e2e8f0">{cov_pct}%</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Clinical Interpretation</h2>
    <p class="interp">{interpretation}</p>
  </div>

  <div class="card">
    <h2>Recommendations</h2>
    {recs_html}
  </div>

  <p style="margin-top:32px;font-size:11px;color:#475569;text-align:center">
    Generated by Automated Eligibility Screener · {eval_date}
  </p>
</div>
</body>
</html>"""

        logger.info("✓ HTML report generated ({} chars)", len(html))
        return html

    def generate_json_report(self, metrics: AccuracyMetrics) -> dict:
        return metrics.model_dump()


report_generator = AccuracyReportGenerator()
