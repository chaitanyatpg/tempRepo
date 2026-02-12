"""
margin_anomaly_routes_fastapi.py
================================
FastAPI router for the daily anomaly detection pipeline.

Location: flask_app/app/margin_anomaly_routes_fastapi.py

The endpoint mirrors the Flask outliers_daily_run but accepts
an optional run_date (yyyymmdd) and uses Pydantic for validation.

TODO: Add your imports from routes.py / other modules as needed.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# TODO: Add your imports here
# Example:
#   from app.routes import (
#       _run_id, _now_ts, _parse_bool, _get_env_list, _shap_params_from_p,
#       MODE_HANDLERS, _RULES_MODES,
#       getCurrentDODDataframe, resolve_model_path,
#       build_download_url, build_center_output_dir, download_url_for,
#       select_top3_accounts_for_center, generate_account_overview_png,
#       fetch_accounts_history_df_batch, write_run_summary_html,
#       send_email_html, plotter_service,
#   )
#   from app.dailyrunschema import CenterSummary as CtrSum, RunOverview, AccountAnomaly
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(
    prefix="/outliers",
    tags=["Outlier Detection (Daily)"],
)


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class DailyRunRequest(BaseModel):
    """Payload for the daily anomaly runner."""

    run_date: Optional[str] = Field(
        None,
        description="Date to run detection for in yyyymmdd format. If blank, uses latest available COB date.",
        pattern=r"^\d{8}$",
        json_schema_extra={"example": "20260212"},
    )
    centers: Optional[List[str]] = Field(
        None,
        description="Filter to these centers (order doesn't matter)",
        json_schema_extra={"example": ["NPM", "RUM", "IPB", "BPI"]},
    )
    modes: Optional[List[str]] = Field(
        None,
        description="Which modes/handlers to run; defaults to DAILY_MODES from .env",
        json_schema_extra={"example": ["aiv7", "aiv8"]},
    )
    top_n: Optional[int] = Field(
        None,
        description="How many items handlers include in their JSON payload",
        json_schema_extra={"example": 3},
    )
    email: Optional[bool] = Field(
        True,
        description="Whether to send summary email",
    )
    force: Optional[bool] = Field(
        False,
        description="Reserved for idempotency; re-run even if already processed",
    )


class ModeResultResponse(BaseModel):
    success: bool
    center: Optional[str] = None
    mode: Optional[str] = None
    output_file: Optional[str] = None
    csv_url: Optional[str] = None
    flagged_count: Optional[int] = None
    total_count: Optional[int] = None
    error: Optional[str] = None
    note: Optional[str] = None

    model_config = {"extra": "allow"}


class CenterSummaryResponse(BaseModel):
    flagged: int = 0
    total: int = 0
    by_mode: Dict[str, int] = {}


class RunSummaryResponse(BaseModel):
    by_center: Dict[str, CenterSummaryResponse] = {}
    by_mode: Dict[str, int] = {}
    total_flagged: int = 0
    duration_sec: float = 0.0


class DailyRunResponse(BaseModel):
    success: bool
    run_id: str
    run_date: Optional[str] = None
    centers_requested: List[str]
    centers_run: List[str] = []
    modes_run: List[str] = []
    cob_dates_by_center: Dict[str, str] = {}
    results: List[ModeResultResponse] = []
    summary: RunSummaryResponse = RunSummaryResponse()
    summary_html: Optional[str] = None
    email_status: str = "pending"


class ErrorResponse(BaseModel):
    success: bool = False
    run_id: Optional[str] = None
    error: str


# ---------------------------------------------------------------------------
# POST /outliers/daily-anomaly-runner
# ---------------------------------------------------------------------------
@router.post(
    "/daily-anomaly-runner",
    response_model=DailyRunResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request or config error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Daily Anomaly Runner",
)
async def daily_anomaly_runner(body: DailyRunRequest = DailyRunRequest()):
    """
    Daily Anomaly Runner
    --------------------
    Runs the margin-anomaly detection pipeline across requested centers
    and modes.

    - If **run_date** is provided (yyyymmdd), it is passed through for
      use in data fetching / output paths.
    - If omitted, the pipeline uses the latest available COB date.
    """
    t0 = time.time()
    run_id = _run_id()
    ts = _now_ts()

    # ---- run_date: use as-is if provided, else None -----------------------
    run_date = body.run_date  # None or "20260212" etc.

    # ---- Defaults from env (overridable by request) -----------------------
    default_centers = _get_env_list("CENTERS", "NPM,RUM,IPB,BPI")
    default_modes = _get_env_list("DAILY_MODES", "aiv7,aiv8")
    default_top_n = int(os.getenv("TOP_N", "20"))
    out_root = os.getenv(
        "OUTLIER_OUTPUT_DIR",
        r"C:\Git\pythonML\1062025\flask_app\out",
    )

    # ---- Request overrides ------------------------------------------------
    requested_centers = body.centers or default_centers
    requested_modes = body.modes or default_modes
    top_n = body.top_n if body.top_n is not None else default_top_n
    email_enabled = body.email if body.email is not None else True
    _ = body.force if body.force is not None else False

    # ---- Sanity: ensure unique, stable order ------------------------------
    centers = list(dict.fromkeys([c.upper() for c in requested_centers]))
    modes = list(dict.fromkeys([m.lower() for m in requested_modes]))

    try:
        # 1) Fetch single DF for all centers --------------------------------
        full_df = getCurrentDODDataframe(centers)

        # Quick validations
        if not isinstance(full_df, pd.DataFrame) or full_df.empty:
            return DailyRunResponse(
                success=True,
                run_id=run_id,
                run_date=run_date,
                centers_requested=centers,
                modes_run=modes,
                results=[],
                summary=RunSummaryResponse(),
                email_status="skipped",
            )

        # Coerce essential fields
        if "MARGIN_CENTER" not in full_df.columns:
            raise HTTPException(
                status_code=500,
                detail={"success": False, "error": "DataFrame missing MARGIN_CENTER column"},
            )
        if "BUSINESS_DATE" not in full_df.columns:
            raise HTTPException(
                status_code=500,
                detail={"success": False, "error": "DataFrame missing BUSINESS_DATE column"},
            )

        # Normalize BUSINESS_DATE to datetime
        full_df["BUSINESS_DATE"] = pd.to_datetime(
            full_df["BUSINESS_DATE"], errors="coerce"
        )
        full_df = full_df.dropna(subset=["BUSINESS_DATE"])

        # Centers actually present
        centers_present = [
            c for c in centers
            if (full_df["MARGIN_CENTER"].str.upper() == c).any()
        ]
        if not centers_present:
            return DailyRunResponse(
                success=True,
                run_id=run_id,
                run_date=run_date,
                centers_requested=centers,
                modes_run=modes,
                results=[],
                summary=RunSummaryResponse(),
                email_status="skipped",
            )

        # ---- Accumulators -------------------------------------------------
        results: List[Dict[str, Any]] = []
        cob_by_center: Dict[str, str] = {}
        summary_by_center: Dict[str, Any] = {}
        summary_by_mode: Dict[str, int] = {m: 0 for m in modes}

        date_root_dir: Optional[str] = None
        per_center_results_map: Dict[str, list] = {c: [] for c in centers}

        # 2) Split by center and compute COB per center ---------------------
        for center in centers_present:
            center_mask = full_df["MARGIN_CENTER"].str.upper() == center
            center_df = full_df.loc[center_mask].copy()

            if center_df.empty:
                results.append({
                    "success": True,
                    "center": center,
                    "mode": None,
                    "flagged_count": 0,
                    "total_count": 0,
                    "note": "No rows for center; skipped.",
                })
                continue

            cob_dt = center_df["BUSINESS_DATE"].max()
            cob_str = cob_dt.strftime("%Y%m%d")
            cob_by_center[center] = cob_dt.strftime("%Y-%m-%d")

            # Establish date root once per run
            if date_root_dir is None:
                date_root_dir = os.path.join(out_root, cob_str)
                os.makedirs(date_root_dir, exist_ok=True)

            # Build base output dir: <root>/<COB_YYYYMMDD>/<CENTER>
            try:
                center_out_dir = build_center_output_dir(out_root, cob_str, center)
            except NotImplementedError:
                center_out_dir = os.path.join(out_root, cob_str, center)
                os.makedirs(center_out_dir, exist_ok=True)

            # Per-center counters
            center_flagged_total = 0
            center_overall_total = 0
            center_flagged_by_mode: Dict[str, int] = {}

            # 3) Run selected modes for this center -------------------------
            for mode in modes:
                handler = MODE_HANDLERS.get(mode)
                if handler is None:
                    results.append({
                        "success": False,
                        "center": center,
                        "mode": mode,
                        "error": f"Invalid mode: {mode}",
                    })
                    continue

                # Build output file path
                base_name = f"{mode}_{center}_{ts}_out.csv"
                out_csv = os.path.join(center_out_dir, base_name)

                # Model path from env (per mode+center)
                try:
                    model_path = resolve_model_path(mode, center)
                except FileNotFoundError as ex:
                    results.append({
                        "success": False,
                        "center": center,
                        "mode": mode,
                        "error": str(ex),
                    })
                    continue

                # Build handler params
                shap_enabled = _parse_bool(os.getenv("SHAP_ENABLED", "true"), True)
                params = {
                    "mode": mode,
                    "center": center,
                    "action": "analyze",
                    "input_df": center_df,
                    "out_csv": out_csv,
                    "top_n": top_n,
                }
                if model_path:
                    params["model_path"] = model_path

                params["shap"] = shap_enabled
                params["shap_top_k"] = int(os.getenv("SHAP_TOP_K", "50"))
                params["shap_dominance_threshold"] = float(
                    os.getenv("SHAP_DOMINANCE_THRESHOLD", "0.5")
                )
                params["shap_print_top"] = int(os.getenv("SHAP_PRINT_TOP", "3"))

                # Invoke handler
                try:
                    result = handler(params)
                    if isinstance(result, tuple) and len(result) == 2:
                        result = result[0]  # normalize (dict, status) → dict
                except Exception as ex:
                    results.append({
                        "success": False,
                        "center": center,
                        "mode": mode,
                        "error": f"Handler error: {ex.__class__.__name__}: {ex}",
                    })
                    continue

                # Normalize/augment per-mode result
                if isinstance(result, dict):
                    result.setdefault("center", center)
                    result.setdefault("mode", mode)
                    result.setdefault("output_file", out_csv)

                    try:
                        csv_url = build_download_url(out_csv)
                    except NotImplementedError:
                        csv_url = None
                    if csv_url:
                        result.setdefault("csv_url", csv_url)

                    flagged = int(result.get("flagged_count", 0) or 0)
                    total = int(result.get("total_count", 0) or 0)
                    center_flagged_total += flagged
                    center_overall_total += total
                    summary_by_mode[mode] = summary_by_mode.get(mode, 0) + flagged
                    center_flagged_by_mode[mode] = center_flagged_by_mode.get(mode, 0) + flagged

                    results.append(result)
                    per_center_results_map[center].append(result)
                else:
                    results.append({
                        "success": False,
                        "center": center,
                        "mode": mode,
                        "error": "Handler returned unexpected result type",
                    })

            # Save per-center summary counters
            summary_by_center[center] = {
                "flagged": center_flagged_total,
                "total": center_overall_total,
                "by_mode": center_flagged_by_mode,
            }

        # 4) Aggregate run-level summary ------------------------------------
        duration_sec = round(time.time() - t0, 2)
        total_flagged = sum(
            v.get("flagged", 0) for v in summary_by_center.values()
        )

        if date_root_dir is None:
            date_root_dir = os.path.join(
                out_root, datetime.today().strftime("%Y%m%d")
            )
            os.makedirs(date_root_dir, exist_ok=True)

        # Build CenterSummary objects + generate images
        by_center_schema: Dict[str, CtrSum] = {}
        basic_features = plotter_service.feature_config["basic_features"]

        for center, counters in summary_by_center.items():
            top3 = select_top3_accounts_for_center(
                center, per_center_results_map.get(center, [])
            )

            if top3:
                center_out_dir = os.path.join(date_root_dir, center)
                os.makedirs(center_out_dir, exist_ok=True)

                top3_accounts = [str(a.account) for a in top3]
                try:
                    df_map = fetch_accounts_history_df_batch(
                        center=center,
                        accounts=top3_accounts,
                        lookback_days=int(os.getenv("HISTORY_LOOKBACK_DAYS", "90")),
                        end_date=pd.to_datetime(cob_by_center[center]),
                        columns=None,
                        features_for_deltas=basic_features,
                    )
                except Exception:
                    df_map = {a: pd.DataFrame() for a in top3_accounts}

                for a in top3:
                    try:
                        def _hist_loader(_c, acct, _df_map=df_map):
                            return _df_map.get(str(acct), pd.DataFrame())

                        img_path = generate_account_overview_png(
                            plotter_service=plotter_service,
                            history_loader=_hist_loader,
                            center_out_dir=center_out_dir,
                            center=center,
                            account=a.account,
                            shap_reason=a.shap_reason,
                        )
                        a.image_path = img_path
                    except Exception:
                        a.image_path = None

            by_center_schema[center] = CtrSum(
                center=center,
                flagged=counters["flagged"],
                total=counters["total"],
                top_accounts=top3,
                flagged_by_mode=counters.get("by_mode", {}),
            )

        # Create summary HTML
        run_overview = RunOverview(
            run_id=run_id,
            cob_dates_by_center=cob_by_center,
            by_center=by_center_schema,
            by_mode=summary_by_mode,
            total_flagged=total_flagged,
            duration_sec=duration_sec,
            date_root_dir=date_root_dir,
        )

        summary_html = write_run_summary_html(
            run_overview,
            title="Daily Outlier Summary",
            embed_images_as_data_uri=False,
        )

        # Email
        email_status: str
        if email_enabled:
            to_env = os.getenv("OUTLIER_EMAIL_TO", "UAT2@no-collab.barclayscorp.com")
            to_addrs = [x.strip() for x in to_env.split(",") if x.strip()]
            subject = f"Daily Outlier Summary \u2502 {run_id}"

            email_html = write_run_summary_html(
                run_overview,
                title="Daily Outlier Summary",
                embed_images_as_data_uri=True,
            )

            try:
                send_email_html(
                    subject=subject,
                    html_body=email_html,
                    to_addrs=to_addrs,
                )
                email_status = "sent"
            except Exception:
                email_status = "error"
        else:
            email_status = "skipped"

        # ---- Final response -----------------------------------------------
        return DailyRunResponse(
            success=True,
            run_id=run_id,
            run_date=run_date,
            centers_requested=centers,
            centers_run=list(summary_by_center.keys()),
            modes_run=modes,
            cob_dates_by_center=cob_by_center,
            results=[
                ModeResultResponse(**r) if isinstance(r, dict) else r
                for r in results
            ],
            summary=RunSummaryResponse(
                by_center={
                    k: CenterSummaryResponse(**v)
                    for k, v in summary_by_center.items()
                },
                by_mode=summary_by_mode,
                total_flagged=total_flagged,
                duration_sec=duration_sec,
            ),
            summary_html=summary_html,
            email_status=email_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "run_id": run_id,
                "error": f"Internal error: {e.__class__.__name__}: {e}",
            },
        )
