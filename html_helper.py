# helpers_html.py
import os
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

from typing import Dict, List, Any, Callable, Optional
from html import escape
import pandas as pd

from .dailyrunschema import AccountAnomaly, CenterSummary, RunOverview


def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))[:128]


def select_top3_accounts_for_center(center: str,
                                     per_mode_results: List[Dict[str, Any]]) -> List[AccountAnomaly]:
    """
    Collapse all modes for a center → best 3 accounts by highest anomaly_score.
    Deduplicate by 'header'.
    per_mode_results[i] can contain key 'top_outliers': List[dict]
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in per_mode_results:
        mode = r.get("mode")
        for o in (r.get("top_outliers") or []):
            acct = str(o.get("header") or "N/A")
            score = float(o.get("anomaly_score") or 0.0)
            shap_reason = o.get("shap_reason")
            prev = best.get(acct)
            if prev is None or score > prev["anomaly_score"]:
                best[acct] = {"anomaly_score": score, "shap_reason": shap_reason, "mode": mode}
    ranked = sorted(best.items(), key=lambda kv: kv[1]["anomaly_score"], reverse=True)[:3]
    return [
        AccountAnomaly(
            account=a,
            anomaly_score=v["anomaly_score"],
            shap_reason=v.get("shap_reason"),
            mode=v.get("mode"),
        ) for a, v in ranked
    ]


def generate_account_overview_png(plotter_service,
                                   history_loader: Callable[[str, str], pd.DataFrame],
                                   center_out_dir: str,
                                   center: str,
                                   account: str,
                                   shap_reason: str | None = None) -> str:
    """
    Generates the combined overview plot and ensures it is saved as:
        <center_out_dir>/<accountID>_Combined_overview.png
    """
    os.makedirs(center_out_dir, exist_ok=True)
    df_hist = history_loader(center, account)

    fname = f"{safe_slug(account)}_Combined_overview.png"
    generated = plotter_service.plot_combined_overview2(
        df_hist,
        center_out_dir,
        identifier=f"{center} │ {account}",
        shap_reason=shap_reason,
        filename=fname
    )

    target_name = f"{safe_slug(account)}_Combined_overview.png"
    out_path = os.path.join(center_out_dir, target_name)

    return out_path


def image_to_data_uri(img_path: str) -> str:
    with open(img_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Compute UNIQUE account counts per center (fixes double-counting)
# ---------------------------------------------------------------------------
def _compute_unique_accounts_per_center(run: RunOverview) -> Dict[str, int]:
    """
    Returns the number of UNIQUE accounts (by header) per center,
    across all modes. This prevents double-counting when multiple
    modes process the same set of accounts.
    """
    unique_per_center: Dict[str, set] = {}
    for c, cs in run.by_center.items():
        seen = set()
        if hasattr(cs, 'top_accounts') and cs.top_accounts:
            for acc in cs.top_accounts:
                seen.add(str(acc.account))
        unique_per_center[c] = seen
    # We can't reconstruct full unique accounts from top_outliers alone,
    # so we use cs.total which should come from the input DF row count
    # per center (already unique). The real fix: ensure cs.total is set
    # from len(center_df) in the endpoint, NOT summed across modes.
    return {c: len(s) if s else 0 for c, s in unique_per_center.items()}


# ===================================================================
# SUMMARY HTML (full detailed report saved to disk)
# ===================================================================
def write_run_summary_html(run: RunOverview,
                            title: str = "Daily Outlier Summary",
                            embed_images_as_data_uri: bool = False) -> str:
    """
    Generate <date_root_dir>/summary.html or return HTML string for email.

    - embed_images_as_data_uri=True:  inline base64 thumbnail with link to full network path
    - embed_images_as_data_uri=False: disk HTML uses <details> to collapse images by default,
      plus an 'Open image' link (href to PNG).
    """
    # Single COB label if uniform across centers
    unique_cobs = sorted(set((run.cob_dates_by_center or {}).values()))
    cob_text = unique_cobs[0] if len(unique_cobs) == 1 else None

    css = """
<style>
  :root {
    --bg: #ffffff; --text: #1a1a2e; --muted: #6b7280; --border: #e5e7eb;
    --accent: #2563eb; --accent-light: #dbeafe;
    --chip-bg: #f0f9ff; --chip-fg: #1e40af;
    --kpi-bg: #f8fafc;
    --success: #059669; --warning: #d97706; --danger: #dc2626;
    --card-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; margin: 0; padding: 0 }
  body {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.5; color: var(--text); background: var(--bg);
    max-width: 1100px; margin: 0 auto; padding: 32px 24px;
  }

  /* Header */
  .report-header { margin-bottom: 32px; border-bottom: 2px solid var(--accent); padding-bottom: 16px }
  .report-header h1 { font-size: 22px; font-weight: 700; color: var(--text) }
  .report-header .meta { color: var(--muted); font-size: 13px; margin-top: 4px }

  /* KPI strip */
  .kpis {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 20px 0 28px 0;
  }
  .kpi {
    background: var(--kpi-bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; text-align: center;
  }
  .kpi .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px }
  .kpi .value { font-size: 20px; font-weight: 700; margin-top: 2px }
  .kpi .value.flagged { color: var(--danger) }

  /* Section headers */
  h2 {
    font-size: 15px; font-weight: 600; color: var(--text);
    border-bottom: 1px solid var(--border); padding-bottom: 6px;
    margin: 28px 0 12px 0;
  }

  /* Tables */
  table { width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 13px }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border) }
  th { background: #f9fafb; font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px }
  tr:hover td { background: #f8fafc }

  /* Chips */
  .chip {
    display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px;
    background: var(--chip-bg); color: var(--chip-fg); font-weight: 500;
    margin-right: 4px;
  }

  /* Rate bar */
  .rate-bar { display: flex; align-items: center; gap: 8px }
  .rate-bar-fill { height: 6px; border-radius: 3px; background: var(--accent); min-width: 2px }
  .rate-bar-track { flex: 1; height: 6px; border-radius: 3px; background: var(--border) }
  .rate-text { font-size: 12px; font-weight: 600; min-width: 42px; text-align: right }

  /* Account cards */
  .center-section { margin-top: 24px }
  .acct-card {
    border: 1px solid var(--border); border-radius: 8px; padding: 16px;
    margin-bottom: 12px; box-shadow: var(--card-shadow);
  }
  .acct-rank { display: inline-block; width: 24px; height: 24px; border-radius: 50%; background: var(--accent); color: white; text-align: center; line-height: 24px; font-size: 12px; font-weight: 700; margin-right: 8px }
  .acct-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px }
  .acct-header strong { font-size: 14px }
  .acct-meta { display: flex; gap: 16px; font-size: 12px; color: var(--muted) }
  .acct-meta span { display: flex; align-items: center; gap: 4px }
  .shap-box { margin-top: 8px; padding: 8px 12px; background: #fffbeb; border-left: 3px solid var(--warning); border-radius: 4px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; color: #92400e }

  /* Image handling */
  .img-section { margin-top: 12px }
  img.preview { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px }
  img.thumb { max-width: 480px }
  .btn {
    display: inline-block; padding: 5px 12px; font-size: 12px; border-radius: 6px;
    border: 1px solid var(--accent); background: white; color: var(--accent);
    text-decoration: none; cursor: pointer; margin-top: 6px;
  }
  .btn:hover { background: var(--accent-light) }

  details > summary {
    cursor: pointer; color: var(--accent); font-size: 12px; font-weight: 500;
    padding: 4px 0; list-style: none; user-select: none;
  }
  details > summary::-webkit-details-marker { display: none }
  details > summary::before { content: '▸ '; display: inline }
  details[open] > summary::before { content: '▾ ' }
  .clamped { max-height: 70vh; object-fit: contain }
  .muted { color: var(--muted); font-size: 12px }

  /* Footer */
  .footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11px; text-align: center }
</style>
"""

    # Build HTML
    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>{css}</head><body>"]

    # Header
    html.append("<div class='report-header'>")
    html.append(f"<h1>{escape(title)}</h1>")
    meta_parts = [f"<strong>Run ID:</strong> {escape(run.run_id)}"]
    if cob_text:
        meta_parts.append(f"<strong>COB:</strong> {escape(cob_text)}")
    elif run.cob_dates_by_center:
        cob_list = ", ".join(f"{k}: {v}" for k, v in sorted(run.cob_dates_by_center.items()))
        meta_parts.append(f"<strong>COB:</strong> {escape(cob_list)}")
    html.append(f"<div class='meta'>{' &nbsp;│&nbsp; '.join(meta_parts)}</div>")
    html.append("</div>")

    # KPIs
    html.append("<div class='kpis'>")
    html.append(f"<div class='kpi'><div class='label'>Total Flagged</div><div class='value flagged'>{run.total_flagged}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Centers</div><div class='value'>{len(run.by_center)}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Modes</div><div class='value'>{len(run.by_mode) if run.by_mode else 0}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Duration</div><div class='value'>{run.duration_sec:.1f}s</div></div>")
    html.append("</div>")

    # Flagged by Mode table
    if getattr(run, "by_mode", None):
        html.append("<h2>Flagged by Mode</h2>")
        html.append("<table><thead><tr><th>Mode</th><th>Flagged</th><th>Share</th></tr></thead><tbody>")
        total_flagged = int(getattr(run, "total_flagged", 0) or 0)
        for mode in sorted(run.by_mode.keys()):
            cnt = int(run.by_mode.get(mode) or 0)
            share = (cnt / total_flagged * 100.0) if total_flagged > 0 else 0.0
            html.append(f"<tr><td><span class='chip'>{escape(mode)}</span></td><td>{cnt}</td><td>{share:.1f}%</td></tr>")
        html.append("</tbody></table>")

    # Flagged by Center table — FIXED: use cs.total directly (unique per center DF)
    html.append("<h2>Flagged by Center</h2>")
    html.append("<table><thead><tr>"
                "<th>Center</th><th>Total Accounts</th><th>Flagged</th><th>Rate</th><th>By Mode</th>"
                "</tr></thead><tbody>")

    for c in sorted(run.by_center.keys()):
        cs: CenterSummary = run.by_center[c]
        # cs.total should be the unique account count from center_df
        # NOT the sum across modes (this was the double-counting bug)
        total = int(cs.total or 0)
        flagged = int(cs.flagged or 0)
        rate = (flagged / total * 100.0) if total > 0 else 0.0

        # Rate bar visual
        rate_html = (
            f"<div class='rate-bar'>"
            f"<div class='rate-bar-track'><div class='rate-bar-fill' style='width:{min(rate, 100):.0f}%'></div></div>"
            f"<div class='rate-text'>{rate:.1f}%</div>"
            f"</div>"
        )

        # chips
        chips = ""
        if getattr(cs, "flagged_by_mode", None):
            parts = []
            for m in sorted(cs.flagged_by_mode.keys()):
                cnt = int(cs.flagged_by_mode.get(m) or 0)
                parts.append(f"<span class='chip'>{escape(m)}: {cnt}</span>")
            chips = " ".join(parts)
        else:
            chips = "<span class='muted'>n/a</span>"

        html.append(
            f"<tr><td><strong>{escape(cs.center)}</strong></td><td>{total}</td>"
            f"<td>{flagged}</td><td>{rate_html}</td><td>{chips}</td></tr>"
        )

    html.append("</tbody></table>")

    # Per-center top 3 account cards
    for c in sorted(run.by_center.keys()):
        cs = run.by_center[c]
        html.append(f"<div class='center-section'>")
        html.append(f"<h2>{escape(c)} — Top Anomaly Accounts</h2>")

        if not cs.top_accounts:
            html.append("<div class='muted'>No anomalies found.</div>")
            html.append("</div>")
            continue

        for idx, acc in enumerate(cs.top_accounts, start=1):
            html.append("<div class='acct-card'>")

            # Header row: rank + account + score
            html.append("<div class='acct-header'>")
            html.append(f"<span class='acct-rank'>{idx}</span>")
            html.append(f"<strong>{escape(str(acc.account))}</strong>")
            html.append("</div>")

            # Meta row: score + mode
            html.append("<div class='acct-meta'>")
            html.append(f"<span>Score: <strong>{float(acc.anomaly_score):.3f}</strong></span>")
            mode_html = f"<span>Mode: {escape(acc.mode)}</span>" if getattr(acc, "mode", None) else ""
            html.append(mode_html)
            html.append("</div>")

            # SHAP reason
            if getattr(acc, "shap_reason", None):
                html.append(f"<div class='shap-box'>💡 {escape(acc.shap_reason)}</div>")

            # Image handling
            img_path = getattr(acc, "image_path", None)
            if img_path and os.path.exists(img_path):
                if embed_images_as_data_uri:
                    thumb_uri = _image_to_data_uri_thumbnail(img_path, max_w=480, quality=70)
                    full_href = _convert_to_file_url(img_path)
                    if thumb_uri:
                        html.append(
                            "<div class='img-section'>"
                            f"<a href='{escape(full_href)}' target='_blank'>"
                            f"<img class='preview thumb' alt='Overview {escape(str(acc.account))}' src='{escape(thumb_uri)}' />"
                            "</a>"
                            "</div>"
                        )
                        html.append(
                            "<div>"
                            f"<a class='btn' href='{escape(full_href)}' target='_blank'>View full resolution</a>"
                            "</div>"
                        )
                    else:
                        html.append("<div class='muted'>[Thumbnail generation failed]</div>")
                else:
                    # WEB MODE: Collapsible details
                    rel = os.path.relpath(img_path, start=run.date_root_dir)
                    html.append(
                        "<div>"
                        f"<a class='btn' href='{escape(rel)}' target='_blank'>Open image</a>"
                        "</div>"
                    )
                    html.append("<details>")
                    html.append("<summary>Show image preview</summary>")
                    html.append(
                        "<div class='img-section'>"
                        f"<img class='preview clamped' loading='lazy' "
                        f"alt='Overview {escape(str(acc.account))}' src='{escape(rel)}' />"
                        "</div>"
                    )
                    html.append("</details>")
            else:
                html.append("<div class='muted'>No overview image.</div>")

            html.append("</div>")  # .acct-card

        html.append("</div>")  # .center-section

    # Footer
    html.append(f"<div class='footer'>Generated by Margin Anomaly Detection &nbsp;│&nbsp; Run {escape(run.run_id)}</div>")
    html.append("</body></html>")

    full_html = "\n".join(html)

    if not embed_images_as_data_uri:
        out_path = os.path.join(run.date_root_dir, "summary.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        return out_path
    else:
        return full_html


# ===================================================================
# EMAIL HTML (concise summary with link to full report)
# ===================================================================
def write_email_html(run: RunOverview,
                     summary_html_path: Optional[str] = None,
                     summary_html_url: Optional[str] = None,
                     title: str = "Daily Outlier Summary") -> str:
    """
    Generate a concise, email-friendly HTML summary.

    This is meant to be the email BODY — clean, scannable, low-noise.
    The full detailed report (with images) lives in summary.html and
    is linked from the email.

    Parameters
    ----------
    run : RunOverview
        The run data.
    summary_html_path : str, optional
        Local file path to the summary.html (for file:// link).
    summary_html_url : str, optional
        Network/web URL to the summary.html (preferred over path).
    title : str
        Email heading.

    Returns
    -------
    str : the email HTML string
    """
    unique_cobs = sorted(set((run.cob_dates_by_center or {}).values()))
    cob_text = unique_cobs[0] if len(unique_cobs) == 1 else ", ".join(unique_cobs)

    # Determine the link to the full report
    report_link = None
    if summary_html_url:
        report_link = summary_html_url
    elif summary_html_path:
        report_link = _convert_to_file_url(summary_html_path)

    # Build center rows for the table
    center_rows = ""
    for c in sorted(run.by_center.keys()):
        cs = run.by_center[c]
        total = int(cs.total or 0)
        flagged = int(cs.flagged or 0)
        rate = (flagged / total * 100.0) if total > 0 else 0.0

        # Top account for this center (just #1)
        top_acct = ""
        if cs.top_accounts:
            a = cs.top_accounts[0]
            top_acct = f"{a.account} ({a.anomaly_score:.2f})"

        center_rows += f"""
        <tr>
            <td style="padding:10px 14px; border-bottom:1px solid #e5e7eb; font-weight:600">{escape(c)}</td>
            <td style="padding:10px 14px; border-bottom:1px solid #e5e7eb; text-align:center">{total}</td>
            <td style="padding:10px 14px; border-bottom:1px solid #e5e7eb; text-align:center; color:#dc2626; font-weight:600">{flagged}</td>
            <td style="padding:10px 14px; border-bottom:1px solid #e5e7eb; text-align:center">{rate:.1f}%</td>
            <td style="padding:10px 14px; border-bottom:1px solid #e5e7eb; font-size:12px; color:#6b7280">{escape(top_acct)}</td>
        </tr>"""

    # Build mode chips
    mode_chips = ""
    if run.by_mode:
        total_flagged = int(run.total_flagged or 0)
        for mode in sorted(run.by_mode.keys()):
            cnt = int(run.by_mode.get(mode) or 0)
            mode_chips += (
                f'<span style="display:inline-block; padding:3px 10px; margin:2px 4px 2px 0; '
                f'border-radius:9999px; background:#f0f9ff; color:#1e40af; font-size:12px; font-weight:500">'
                f'{escape(mode)}: {cnt}</span>'
            )

    # CTA button
    cta_html = ""
    if report_link:
        cta_html = f"""
        <div style="text-align:center; margin:28px 0">
            <a href="{escape(report_link)}" target="_blank"
               style="display:inline-block; padding:12px 32px; background:#2563eb; color:#ffffff;
                      text-decoration:none; border-radius:8px; font-size:14px; font-weight:600;
                      letter-spacing:0.3px">
                View Full Report →
            </a>
        </div>
        """

    # Top anomalous accounts across all centers (max 5)
    top_accounts_html = ""
    all_top = []
    for c in sorted(run.by_center.keys()):
        cs = run.by_center[c]
        if cs.top_accounts:
            for a in cs.top_accounts:
                all_top.append((c, a))
    # sort by score desc, take top 5
    all_top.sort(key=lambda x: x[1].anomaly_score, reverse=True)
    all_top = all_top[:5]

    if all_top:
        rows = ""
        for center, acc in all_top:
            reason = escape(acc.shap_reason[:60] + "…") if acc.shap_reason and len(acc.shap_reason) > 60 else escape(acc.shap_reason or "—")
            rows += f"""
            <tr>
                <td style="padding:8px 12px; border-bottom:1px solid #e5e7eb; font-size:13px">{escape(center)}</td>
                <td style="padding:8px 12px; border-bottom:1px solid #e5e7eb; font-size:13px; font-weight:600">{escape(str(acc.account))}</td>
                <td style="padding:8px 12px; border-bottom:1px solid #e5e7eb; font-size:13px; text-align:center">{acc.anomaly_score:.3f}</td>
                <td style="padding:8px 12px; border-bottom:1px solid #e5e7eb; font-size:12px; color:#92400e; font-family:monospace">{reason}</td>
            </tr>"""

        top_accounts_html = f"""
        <div style="margin-top:24px">
            <h3 style="font-size:14px; font-weight:600; color:#1a1a2e; margin-bottom:8px">🔍 Top Anomalous Accounts</h3>
            <table style="width:100%; border-collapse:collapse">
                <thead>
                    <tr style="background:#f9fafb">
                        <th style="padding:8px 12px; text-align:left; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Center</th>
                        <th style="padding:8px 12px; text-align:left; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Account</th>
                        <th style="padding:8px 12px; text-align:center; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Score</th>
                        <th style="padding:8px 12px; text-align:left; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Reason</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    email_html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0; padding:0; background:#f3f4f6; font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif">

<!-- Outer container -->
<div style="max-width:640px; margin:24px auto; background:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06)">

    <!-- Blue header bar -->
    <div style="background:#1e3a5f; padding:24px 28px; color:#ffffff">
        <h1 style="margin:0; font-size:20px; font-weight:700">{escape(title)}</h1>
        <div style="margin-top:6px; font-size:13px; opacity:0.85">
            COB: {escape(cob_text)} &nbsp;│&nbsp; Run: {escape(run.run_id)}
        </div>
    </div>

    <!-- Body -->
    <div style="padding:24px 28px">

        <!-- KPI strip -->
        <div style="display:flex; gap:12px; margin-bottom:24px">
            <div style="flex:1; text-align:center; padding:16px 12px; background:#fef2f2; border-radius:8px; border:1px solid #fecaca">
                <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px">Total Flagged</div>
                <div style="font-size:28px; font-weight:800; color:#dc2626; margin-top:4px">{run.total_flagged}</div>
            </div>
            <div style="flex:1; text-align:center; padding:16px 12px; background:#f0fdf4; border-radius:8px; border:1px solid #bbf7d0">
                <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px">Centers</div>
                <div style="font-size:28px; font-weight:800; color:#059669; margin-top:4px">{len(run.by_center)}</div>
            </div>
            <div style="flex:1; text-align:center; padding:16px 12px; background:#f8fafc; border-radius:8px; border:1px solid #e5e7eb">
                <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px">Duration</div>
                <div style="font-size:28px; font-weight:800; color:#1a1a2e; margin-top:4px">{run.duration_sec:.0f}s</div>
            </div>
        </div>

        <!-- Mode chips -->
        <div style="margin-bottom:20px">
            <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px">Modes Run</div>
            {mode_chips}
        </div>

        <!-- Center summary table -->
        <table style="width:100%; border-collapse:collapse; margin-bottom:8px">
            <thead>
                <tr style="background:#f9fafb">
                    <th style="padding:10px 14px; text-align:left; font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.3px; border-bottom:2px solid #e5e7eb">Center</th>
                    <th style="padding:10px 14px; text-align:center; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Total</th>
                    <th style="padding:10px 14px; text-align:center; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Flagged</th>
                    <th style="padding:10px 14px; text-align:center; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Rate</th>
                    <th style="padding:10px 14px; text-align:left; font-size:11px; color:#6b7280; text-transform:uppercase; border-bottom:2px solid #e5e7eb">Top Account</th>
                </tr>
            </thead>
            <tbody>{center_rows}</tbody>
        </table>

        {top_accounts_html}

        {cta_html}

    </div>

    <!-- Footer -->
    <div style="padding:16px 28px; background:#f9fafb; border-top:1px solid #e5e7eb; text-align:center; font-size:11px; color:#9ca3af">
        Margin Anomaly Detection &nbsp;│&nbsp; Run {escape(run.run_id)} &nbsp;│&nbsp; This is an automated notification
    </div>

</div>

</body>
</html>"""

    return email_html


# ===================================================================
# Helper utilities
# ===================================================================
def _convert_to_file_url(path: str) -> str:
    """
    Convert a file path to a file:// URL.
    Handles both Windows UNC paths and regular paths.

    Examples:
        \\\\server\\share\\file.png -> file://server/share/file.png
        C:\\folder\\file.png -> file:///C:/folder/file.png
        /mnt/share/file.png -> file:///mnt/share/file.png
    """
    p = Path(path)
    path_str = p.as_posix()

    # Handle UNC paths (\\server\share)
    if path_str.startswith('//'):
        return 'file://' + path_str.lstrip('/')

    # Handle absolute paths
    if p.is_absolute():
        return 'file:///' + path_str.lstrip('/')

    # Relative paths - return as-is
    return path_str


def _image_to_data_uri_thumbnail(path: str, max_w: int = 500, quality: int = 70) -> str:
    """
    Create a reasonably small base64 data URI for email previews.
    JPEG preferred for size; keeps transparency if PNG and small.
    """
    try:
        im = Image.open(path)
        im.load()
        w, h = im.size
        if w > max_w:
            ratio = max_w / float(w)
            im = im.resize((max_w, int(h * ratio)), Image.LANCZOS)

        # Choose format
        fmt = "PNG" if (im.mode in ("RGBA", "LA") or path.lower().endswith(".png")) else "JPEG"
        buf = BytesIO()
        if fmt == "JPEG":
            if im.mode in ("RGBA", "LA"):
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
            mime = "image/jpeg"
        else:
            im.save(buf, format="PNG", optimize=True)
            mime = "image/png"

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""  # fail safe; caller decides fallback
