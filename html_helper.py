# helpers_html.py
import os
import re
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

from typing import Dict, List, Any, Callable, Optional
from html import escape
import pandas as pd

from .dailyrunschema import AccountAnomaly, CenterSummary, RunOverview


# ===================================================================
# Shared utilities
# ===================================================================

def safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))[:128]


def _extract_reason_columns(shap_reason: Optional[str]) -> str:
    """
    Extract just the column names from a SHAP reason string.

    Input:  "DELTA_APPLIED_REQ +0.63, PME +0.22"
    Output: "DELTA_APPLIED_REQ, PME"

    Input:  "GROSS_MV_CHANGE +0.81"
    Output: "GROSS_MV_CHANGE"

    Input:  None or ""
    Output: ""
    """
    if not shap_reason:
        return ""
    parts = []
    for chunk in shap_reason.split(","):
        chunk = chunk.strip()
        col = re.sub(r"\s+[+\-]?\d+\.?\d*$", "", chunk).strip()
        if col:
            parts.append(col)
    return ", ".join(parts)


def select_top3_accounts_for_center(center: str,
                                     per_mode_results: List[Dict[str, Any]]) -> List[AccountAnomaly]:
    """
    Collapse all modes for a center -> best 3 accounts by highest anomaly_score.
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
        identifier=f"{center} | {account}",
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


def _convert_to_file_url(path: str) -> str:
    """
    Convert a file path to a file:// URL.
    Handles both Windows UNC paths and regular paths.
    """
    p = Path(path)
    path_str = p.as_posix()

    if path_str.startswith('//'):
        return 'file://' + path_str.lstrip('/')

    if p.is_absolute():
        return 'file:///' + path_str.lstrip('/')

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
        return ""


# ===================================================================
# EMAIL HTML - concise, business-focused, no internal model details
# ===================================================================

def write_email_html(run: RunOverview,
                     summary_html_path: Optional[str] = None,
                     summary_html_url: Optional[str] = None,
                     title: str = "Margin Anomaly Report") -> str:
    """
    Generate a clean, Outlook-compatible, business-focused email body.
    Uses table-based layout with bgcolor attributes for maximum
    Outlook/Exchange compatibility. Barclays blue (#00395D) styling.
    ASCII-only output (safe for WCF SOAP services).
    """
    unique_cobs = sorted(set((run.cob_dates_by_center or {}).values()))
    cob_text = unique_cobs[0] if len(unique_cobs) == 1 else ", ".join(unique_cobs)

    report_link = None
    if summary_html_url:
        report_link = summary_html_url
    elif summary_html_path:
        report_link = _convert_to_file_url(summary_html_path)

    total_accounts = sum(int(cs.total or 0) for cs in run.by_center.values())
    total_flagged = int(run.total_flagged or 0)
    overall_rate = (total_flagged / total_accounts * 100.0) if total_accounts > 0 else 0.0
    num_centers = len(run.by_center)
    plural = "s" if num_centers != 1 else ""

    # Inline font style shorthand
    F = "font-family:Segoe UI,Arial,sans-serif"

    # ---- Center table rows ------------------------------------------------
    center_rows = ""
    alt = False
    centers_sorted = sorted(run.by_center.keys())
    for i, c in enumerate(centers_sorted):
        cs = run.by_center[c]
        total = int(cs.total or 0)
        flagged = int(cs.flagged or 0)
        rate = (flagged / total * 100.0) if total > 0 else 0.0
        bg = ' bgcolor="#f9fafb" style="background-color:#f9fafb;' if (i % 2 == 1) else ' style="'
        last = "" if i < len(centers_sorted) - 1 else ""
        bb = "border-bottom:1px solid #e5e7eb;" if i < len(centers_sorted) - 1 else ""

        center_rows += (
            f'<tr>'
            f'<td{bg} padding:10px 14px; {bb} {F}; font-size:13px; font-weight:600; color:#1a1a2e;">{escape(cs.center)}</td>'
            f'<td{bg} padding:10px 14px; {bb} {F}; font-size:13px; color:#374151; text-align:center;">{total:,}</td>'
            f'<td{bg} padding:10px 14px; {bb} {F}; font-size:13px; color:#dc2626; font-weight:700; text-align:center;">{flagged}</td>'
            f'<td{bg} padding:10px 14px; {bb} {F}; font-size:13px; color:#374151; text-align:center;">{rate:.1f}%</td>'
            f'</tr>'
        )

    # ---- Top 3 per center -------------------------------------------------
    top3_html = ""
    for c in centers_sorted:
        cs = run.by_center[c]

        # Section header with cyan underline
        top3_html += (
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:8px;">'
            f'<tr><td style="{F}; font-size:14px; font-weight:bold; color:#00395D; padding-bottom:8px; border-bottom:3px solid #00AEEF;">'
            f'{escape(c)} - Top 3 Flagged Accounts'
            f'</td></tr></table>'
        )

        if not cs.top_accounts:
            top3_html += (
                f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:24px;">'
                f'<tr><td style="{F}; font-size:12px; color:#9ca3af; padding:8px 0;">No anomalies detected.</td></tr>'
                f'</table>'
            )
            continue

        top3_html += '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:24px;">'

        for idx, acc in enumerate(cs.top_accounts, start=1):
            reason = _extract_reason_columns(getattr(acc, "shap_reason", None))
            reason_cell = ""
            if reason:
                reason_cell = (
                    f'<tr><td style="padding-top:4px;">'
                    f'<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
                    f'<td bgcolor="#FFF3E0" style="background-color:#FFF3E0; padding:3px 10px; {F}; font-size:11px; color:#b45309; border:1px solid #f59e0b;">{escape(reason)}</td>'
                    f'</tr></table>'
                    f'</td></tr>'
                )

            top3_html += (
                f'<tr>'
                # Rank badge
                f'<td width="32" valign="top" style="padding:8px 12px 8px 0;">'
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
                f'<td bgcolor="#00395D" width="24" height="24" style="background-color:#00395D; color:#ffffff; {F}; font-size:11px; font-weight:bold; text-align:center; line-height:24px;">{idx}</td>'
                f'</tr></table>'
                f'</td>'
                # Account + reason
                f'<td valign="top" style="padding:8px 0; border-bottom:1px solid #e5e7eb;">'
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0">'
                f'<tr><td style="{F}; font-size:14px; font-weight:bold; color:#1a1a2e;">{escape(str(acc.account))}</td></tr>'
                f'{reason_cell}'
                f'</table>'
                f'</td>'
                f'</tr>'
            )

        top3_html += '</table>'

    # ---- CTA button -------------------------------------------------------
    cta_html = ""
    if report_link:
        cta_html = (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
            '<tr><td align="center" style="padding:24px 0 8px 0;">'
            '<table role="presentation" cellpadding="0" cellspacing="0" border="0">'
            '<tr>'
            f'<td bgcolor="#00395D" style="background-color:#00395D; padding:14px 40px;">'
            f'<a href="{escape(report_link)}" target="_blank" style="{F}; font-size:14px; font-weight:bold; color:#ffffff; text-decoration:none; display:inline-block;">View Full Report and Charts &gt;</a>'
            '</td>'
            '</tr></table>'
            '</td></tr>'
            f'<tr><td align="center" style="padding-top:8px; {F}; font-size:11px; color:#9ca3af;">'
            'Includes account overview images and detailed breakdowns'
            '</td></tr>'
            '</table>'
        )

    # ---- Assemble ---------------------------------------------------------
    email = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<!--[if !mso]><!--><meta http-equiv="X-UA-Compatible" content="IE=edge"><!--<![endif]-->'
        '</head>'
        f'<body style="margin:0; padding:0; background-color:#f4f4f4; {F};">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f4f4f4;">'
        '<tr><td align="center" style="padding:24px 8px;">'
        '<table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0" style="background-color:#ffffff; border:1px solid #d0d5dd;">'

        # Header
        '<tr><td bgcolor="#00395D" style="background-color:#00395D; padding:28px 32px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td style="{F}; font-size:22px; font-weight:bold; color:#ffffff; letter-spacing:-0.3px;">{escape(title)}</td></tr>'
        f'<tr><td style="{F}; font-size:13px; color:#a3c4e0; padding-top:6px;">COB: {escape(cob_text)}  |  Run: {escape(run.run_id)}</td></tr>'
        '</table></td></tr>'

        # Cyan accent line
        '<tr><td bgcolor="#00AEEF" style="background-color:#00AEEF; height:4px; font-size:0; line-height:0;"> </td></tr>'

        # Body
        '<tr><td style="padding:28px 32px;">'

        # KPI cards
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:28px;"><tr>'
        '<td width="48%" valign="top">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border:2px solid #dc2626;">'
        '<tr><td bgcolor="#fef2f2" style="background-color:#fef2f2; padding:20px 16px; text-align:center;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td style="{F}; font-size:11px; color:#991b1b; text-transform:uppercase; letter-spacing:1px; font-weight:bold; text-align:center;">ACCOUNTS FLAGGED</td></tr>'
        f'<tr><td style="{F}; font-size:36px; font-weight:800; color:#dc2626; text-align:center; padding-top:6px;">{total_flagged}</td></tr>'
        f'<tr><td style="{F}; font-size:12px; color:#991b1b; text-align:center; padding-top:2px;">out of {total_accounts:,}</td></tr>'
        '</table></td></tr></table></td>'
        '<td width="4%"> </td>'
        '<td width="48%" valign="top">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border:2px solid #059669;">'
        '<tr><td bgcolor="#f0fdf4" style="background-color:#f0fdf4; padding:20px 16px; text-align:center;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td style="{F}; font-size:11px; color:#065f46; text-transform:uppercase; letter-spacing:1px; font-weight:bold; text-align:center;">OVERALL FLAG RATE</td></tr>'
        f'<tr><td style="{F}; font-size:36px; font-weight:800; color:#059669; text-align:center; padding-top:6px;">{overall_rate:.1f}%</td></tr>'
        f'<tr><td style="{F}; font-size:12px; color:#065f46; text-align:center; padding-top:2px;">across {num_centers} center{plural}</td></tr>'
        '</table></td></tr></table></td>'
        '</tr></table>'

        # Summary by Center label
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom:6px;">'
        f'<tr><td style="{F}; font-size:14px; font-weight:bold; color:#00395D; padding-bottom:10px;">Summary by Center</td></tr></table>'

        # Center table
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border:1px solid #d0d5dd; margin-bottom:28px;">'
        '<tr>'
        f'<td bgcolor="#00395D" style="background-color:#00395D; padding:10px 14px; {F}; font-size:11px; color:#ffffff; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; border-bottom:2px solid #00AEEF;">CENTER</td>'
        f'<td bgcolor="#00395D" style="background-color:#00395D; padding:10px 14px; {F}; font-size:11px; color:#ffffff; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; text-align:center; border-bottom:2px solid #00AEEF;">ACCOUNTS</td>'
        f'<td bgcolor="#00395D" style="background-color:#00395D; padding:10px 14px; {F}; font-size:11px; color:#ffffff; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; text-align:center; border-bottom:2px solid #00AEEF;">FLAGGED</td>'
        f'<td bgcolor="#00395D" style="background-color:#00395D; padding:10px 14px; {F}; font-size:11px; color:#ffffff; font-weight:bold; text-transform:uppercase; letter-spacing:0.5px; text-align:center; border-bottom:2px solid #00AEEF;">RATE</td>'
        '</tr>'
        f'{center_rows}'
        '</table>'

        # Top 3 sections
        f'{top3_html}'

        # CTA
        f'{cta_html}'

        '</td></tr>'  # end body

        # Footer
        '<tr><td bgcolor="#f0f2f5" style="background-color:#f0f2f5; padding:16px 32px; border-top:1px solid #d0d5dd;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td align="center" style="{F}; font-size:11px; color:#6b7280;">Margin Anomaly Detection  |  This is an automated daily notification</td></tr>'
        '</table></td></tr>'

        '</table>'  # main card
        '</td></tr></table>'  # outer wrapper
        '</body></html>'
    )

    return email


# ===================================================================
# SUMMARY HTML - full detailed report saved to disk
# ===================================================================

def write_run_summary_html(run: RunOverview,
                            title: str = "Daily Outlier Summary",
                            embed_images_as_data_uri: bool = False) -> str:
    """
    Generate <date_root_dir>/summary.html (detailed report with images).

    This is the FULL report linked from the email. It includes:
      - Model breakdowns, per-mode chips
      - Anomaly scores
      - SHAP reason with numeric contributions
      - Account overview images (collapsible or embedded)
    """
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

  .report-header { margin-bottom: 32px; border-bottom: 2px solid var(--accent); padding-bottom: 16px }
  .report-header h1 { font-size: 22px; font-weight: 700; color: var(--text) }
  .report-header .meta { color: var(--muted); font-size: 13px; margin-top: 4px }

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

  h2 {
    font-size: 15px; font-weight: 600; color: var(--text);
    border-bottom: 1px solid var(--border); padding-bottom: 6px;
    margin: 28px 0 12px 0;
  }

  table { width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 13px }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border) }
  th { background: #f9fafb; font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px }
  tr:hover td { background: #f8fafc }

  .chip {
    display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px;
    background: var(--chip-bg); color: var(--chip-fg); font-weight: 500; margin-right: 4px;
  }

  .rate-bar { display: flex; align-items: center; gap: 8px }
  .rate-bar-fill { height: 6px; border-radius: 3px; background: var(--accent); min-width: 2px }
  .rate-bar-track { flex: 1; height: 6px; border-radius: 3px; background: var(--border) }
  .rate-text { font-size: 12px; font-weight: 600; min-width: 42px; text-align: right }

  .center-section { margin-top: 24px }
  .acct-card {
    border: 1px solid var(--border); border-radius: 8px; padding: 16px;
    margin-bottom: 12px; box-shadow: var(--card-shadow);
  }
  .acct-rank {
    display: inline-block; width: 24px; height: 24px; border-radius: 50%;
    background: var(--accent); color: white; text-align: center; line-height: 24px;
    font-size: 12px; font-weight: 700; margin-right: 8px;
  }
  .acct-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px }
  .acct-header strong { font-size: 14px }
  .acct-meta { display: flex; gap: 16px; font-size: 12px; color: var(--muted) }
  .acct-meta span { display: flex; align-items: center; gap: 4px }
  .shap-box {
    margin-top: 8px; padding: 8px 12px; background: #fffbeb;
    border-left: 3px solid var(--warning); border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px; color: #92400e;
  }

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
  details > summary::before { content: '\\25b8 '; display: inline }
  details[open] > summary::before { content: '\\25be ' }
  .clamped { max-height: 70vh; object-fit: contain }
  .muted { color: var(--muted); font-size: 12px }

  .footer {
    margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 11px; text-align: center;
  }
</style>
"""

    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>{escape(title)}</title>{css}</head><body>"]

    # ---- Header -----------------------------------------------------------
    html.append("<div class='report-header'>")
    html.append(f"<h1>{escape(title)}</h1>")
    meta_parts = [f"<strong>Run ID:</strong> {escape(run.run_id)}"]
    if cob_text:
        meta_parts.append(f"<strong>COB:</strong> {escape(cob_text)}")
    elif run.cob_dates_by_center:
        cob_list = ", ".join(f"{k}: {v}" for k, v in sorted(run.cob_dates_by_center.items()))
        meta_parts.append(f"<strong>COB:</strong> {escape(cob_list)}")
    html.append(f"<div class='meta'>{' &nbsp;&#x2502;&nbsp; '.join(meta_parts)}</div>")
    html.append("</div>")

    # ---- KPIs -------------------------------------------------------------
    html.append("<div class='kpis'>")
    html.append(f"<div class='kpi'><div class='label'>Total Flagged</div><div class='value flagged'>{run.total_flagged}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Centers</div><div class='value'>{len(run.by_center)}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Modes</div><div class='value'>{len(run.by_mode) if run.by_mode else 0}</div></div>")
    html.append(f"<div class='kpi'><div class='label'>Duration</div><div class='value'>{run.duration_sec:.1f}s</div></div>")
    html.append("</div>")

    # ---- Flagged by Mode table --------------------------------------------
    if getattr(run, "by_mode", None):
        html.append("<h2>Flagged by Mode</h2>")
        html.append("<table><thead><tr><th>Mode</th><th>Flagged</th><th>Share</th></tr></thead><tbody>")
        total_f = int(getattr(run, "total_flagged", 0) or 0)
        for mode in sorted(run.by_mode.keys()):
            cnt = int(run.by_mode.get(mode) or 0)
            share = (cnt / total_f * 100.0) if total_f > 0 else 0.0
            html.append(f"<tr><td><span class='chip'>{escape(mode)}</span></td><td>{cnt}</td><td>{share:.1f}%</td></tr>")
        html.append("</tbody></table>")

    # ---- Flagged by Center table ------------------------------------------
    html.append("<h2>Flagged by Center</h2>")
    html.append("<table><thead><tr>"
                "<th>Center</th><th>Total Accounts</th><th>Flagged</th><th>Rate</th><th>By Mode</th>"
                "</tr></thead><tbody>")

    for c in sorted(run.by_center.keys()):
        cs: CenterSummary = run.by_center[c]
        total = int(cs.total or 0)
        flagged = int(cs.flagged or 0)
        rate = (flagged / total * 100.0) if total > 0 else 0.0

        rate_html = (
            f"<div class='rate-bar'>"
            f"<div class='rate-bar-track'><div class='rate-bar-fill' style='width:{min(rate, 100):.0f}%'></div></div>"
            f"<div class='rate-text'>{rate:.1f}%</div>"
            f"</div>"
        )

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
            f"<tr><td><strong>{escape(cs.center)}</strong></td><td>{total:,}</td>"
            f"<td>{flagged}</td><td>{rate_html}</td><td>{chips}</td></tr>"
        )

    html.append("</tbody></table>")

    # ---- Per-center top 3 account cards -----------------------------------
    for c in sorted(run.by_center.keys()):
        cs = run.by_center[c]
        html.append("<div class='center-section'>")
        html.append(f"<h2>{escape(c)} &#x2014; Top Anomaly Accounts</h2>")

        if not cs.top_accounts:
            html.append("<div class='muted'>No anomalies found.</div>")
            html.append("</div>")
            continue

        for idx, acc in enumerate(cs.top_accounts, start=1):
            html.append("<div class='acct-card'>")

            html.append("<div class='acct-header'>")
            html.append(f"<span class='acct-rank'>{idx}</span>")
            html.append(f"<strong>{escape(str(acc.account))}</strong>")
            html.append("</div>")

            html.append("<div class='acct-meta'>")
            html.append(f"<span>Score: <strong>{float(acc.anomaly_score):.3f}</strong></span>")
            if getattr(acc, "mode", None):
                html.append(f"<span>Mode: {escape(acc.mode)}</span>")
            html.append("</div>")

            if getattr(acc, "shap_reason", None):
                html.append(f"<div class='shap-box'>&#x1f4a1; {escape(acc.shap_reason)}</div>")

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

    # ---- Footer -----------------------------------------------------------
    html.append(f"<div class='footer'>Generated by Margin Anomaly Detection &nbsp;&#x2502;&nbsp; Run {escape(run.run_id)}</div>")
    html.append("</body></html>")

    full_html = "\n".join(html)

    if not embed_images_as_data_uri:
        out_path = os.path.join(run.date_root_dir, "summary.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        return out_path
    else:
        return full_html
