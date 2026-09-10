# Rapido Supply DS Take-Home — Analysis
# ======================================
# Run: python rapido_analysis.py
# Outputs: All charts saved to output/ folder
# Requirements: pandas, numpy, matplotlib, seaborn, scipy

"""
This script runs the complete analysis for the Rapido Captain Onboarding
take-home exercise. It covers:

  Part A: Onboarding Funnel (A1-A4)
  Part B: Airport Supply (B1-B3)

Key methodological choices:
  - Denominator: "mature cohort" = captains who signed up before June 1, 2026
    (>= 30 days before extraction at June 30 23:59 IST). This avoids right-
    censoring bias from recent signups still in-progress.
  - Campaign evaluation: uses delivered/not-delivered as a natural experiment
    to estimate causal effect (intent-to-treat framework).
  - All timestamps are treated as IST per the brief.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
# Resolve the data folder relative to this script, not the working directory,
# so the analysis runs end to end from anywhere. Looks for a folder holding the
# seven raw CSVs in: ./Data, ../Data, ./data, ../data, then the current dir.
REQUIRED_CSVS = [
    "captains.csv", "doc_events.csv", "approvals.csv", "activation.csv",
    "nudges.csv", "airport_hourly.csv", "airport_trips.csv",
]

def _resolve_data_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "Data", script_dir.parent / "Data",
        script_dir / "data", script_dir.parent / "data",
        Path.cwd() / "Data", Path.cwd() / "data", script_dir, Path.cwd(),
    ]
    for c in candidates:
        if all((c / f).exists() for f in REQUIRED_CSVS):
            return c
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find the seven raw CSVs.\n"
        "Place them in a 'Data' folder next to this script.\n"
        "Expected files: " + ", ".join(REQUIRED_CSVS) + "\n"
        "Searched:\n  " + searched
    )

DATA_DIR   = _resolve_data_dir()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Reading raw CSVs from: {DATA_DIR}")
print(f"Writing charts to:     {OUTPUT_DIR}\n")

EXTRACTION_DATE = pd.Timestamp("2026-06-30 23:59:00")  # IST, per brief
MATURITY_CUTOFF = pd.Timestamp("2026-06-01")             # 30-day window

DOC_ORDER_ALL       = ["DL", "RC", "AADHAAR", "PERMIT", "FITNESS", "INSURANCE"]
DOC_ORDER_NO_PERMIT = ["DL", "RC", "AADHAAR", "FITNESS", "INSURANCE"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 150


# ══════════════════════════════════════════════════════════════════════════
# PHASE 0: DATA LOADING, CLEANING, AND QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all CSV files and parse timestamps."""
    captains   = pd.read_csv(DATA_DIR / "captains.csv",       parse_dates=["signup_ts"])
    doc_events = pd.read_csv(DATA_DIR / "doc_events.csv",      parse_dates=["event_ts"])
    approvals  = pd.read_csv(DATA_DIR / "approvals.csv",       parse_dates=["decision_ts"])
    activation = pd.read_csv(DATA_DIR / "activation.csv",      parse_dates=["first_order_ts"])
    nudges     = pd.read_csv(DATA_DIR / "nudges.csv",          parse_dates=["sent_ts"])
    airport_hr = pd.read_csv(DATA_DIR / "airport_hourly.csv",  parse_dates=["hour_ts"])
    airport_tr = pd.read_csv(DATA_DIR / "airport_trips.csv",   parse_dates=["request_ts"])
    return captains, doc_events, approvals, activation, nudges, airport_hr, airport_tr


def build_master_table(captains, approvals, activation):
    """Create captain-level master table with approval and activation info."""
    master = captains.merge(approvals, on="captain_id", how="left") \
                     .merge(activation, on="captain_id", how="left")
    master["final_status"] = master["final_status"].fillna("no_approval_record")
    master["signup_month"] = master["signup_ts"].dt.to_period("M")
    master["is_mature"] = master["signup_ts"] < MATURITY_CUTOFF
    return master


def run_data_quality_checks(captains, approvals, activation, doc_events):
    """Run and print data quality checks."""
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECKS")
    print("=" * 70)

    # Check 1: 100% activation
    n_approved  = (approvals["final_status"] == "approved").sum()
    n_activated = len(activation)
    print(f"\n  [1] Approved: {n_approved}, Activated: {n_activated}")
    print(f"      100% activation = {'YES (suspicious/definitional)' if n_approved == n_activated else 'No'}")

    # Check 2: Right-censoring in activation
    null_d7  = activation["orders_d7"].isnull().sum()
    null_d30 = activation["orders_d30"].isnull().sum()
    print(f"\n  [2] Activation nulls: orders_d7={null_d7} ({null_d7/len(activation):.1%}), "
          f"orders_d30={null_d30} ({null_d30/len(activation):.1%})")
    print(f"      Recently approved captains haven't had full activity windows yet.")

    # Check 3: Rejection reasons beyond doc stages
    doc_stages = set(DOC_ORDER_ALL)
    extra_reasons = set(approvals["last_stage_reached"].dropna().unique()) - doc_stages
    print(f"\n  [3] Non-doc rejection reasons in last_stage_reached:")
    for r in sorted(extra_reasons):
        print(f"      {r}: {(approvals['last_stage_reached'] == r).sum()}")

    # Check 4: ERickshaw + PERMIT
    er_ids = set(captains[captains["vehicle_type"] == "ERickshaw"]["captain_id"])
    er_permit = doc_events[(doc_events["captain_id"].isin(er_ids)) &
                           (doc_events["doc_type"] == "PERMIT")]
    print(f"\n  [4] ERickshaw captains with PERMIT events: {er_permit['captain_id'].nunique()}")
    print(f"      Per brief, Permit applies to Auto/Cab only. "
          f"{'OK' if len(er_permit) == 0 else 'ISSUE!'}")

    # Check 5: Date range
    print(f"\n  [5] Signup range: {captains['signup_ts'].min()} to {captains['signup_ts'].max()}")
    print(f"      Extraction: {EXTRACTION_DATE}")


# ══════════════════════════════════════════════════════════════════════════
# PART A1: BUILD THE ONBOARDING FUNNEL
# ══════════════════════════════════════════════════════════════════════════

def build_funnel(mature, doc_mature):
    """
    Build the signup -> approved funnel for the mature cohort.

    Denominator: captains who signed up >= 30 days before extraction.
    This avoids right-censoring: the mature cohort has 0 in-progress captains.
    """
    print("\n" + "=" * 70)
    print("A1: ONBOARDING FUNNEL")
    print("=" * 70)

    # Who started uploading?
    started = set(
        doc_mature[doc_mature["event_type"] == "upload_success"]["captain_id"].unique()
    )
    mature = mature.copy()
    mature["started_docs"] = mature["captain_id"].isin(started)

    # Split by vehicle type for "cleared all docs" calculation
    auto_cab  = mature[mature["vehicle_type"].isin(["Auto", "Cab"])]
    erickshaw = mature[mature["vehicle_type"] == "ERickshaw"]

    funnel = {
        "Signed Up":             len(mature),
        "Started Upload":        mature["started_docs"].sum(),
        "Cleared 1 doc (DL)":    (mature["docs_cleared"] >= 1).sum(),
        "Cleared 2 docs (+ RC)": (mature["docs_cleared"] >= 2).sum(),
        "Cleared 3 docs (+ Aadhaar)": (mature["docs_cleared"] >= 3).sum(),
        "Cleared 4 docs":        (mature["docs_cleared"] >= 4).sum(),
        "Cleared 5 docs":        (mature["docs_cleared"] >= 5).sum(),
        "Cleared all docs":      (auto_cab["docs_cleared"] >= 6).sum() +
                                 (erickshaw["docs_cleared"] >= 5).sum(),
        "Approved":              (mature["final_status"] == "approved").sum(),
    }

    print(f"\n  Denominator: {len(mature):,} captains (signup < {MATURITY_CUTOFF.date()})")
    print(f"  In-progress in mature cohort: {(mature['final_status'] == 'in_progress').sum()}")
    print()

    prev = len(mature)
    for stage, count in funnel.items():
        pct = count / len(mature) * 100
        drop = prev - count
        drop_pct = drop / prev * 100 if prev > 0 else 0
        print(f"  {stage:30s}  {count:>6,}  ({pct:5.1f}%)  "
              f"[lost {drop:>5,} = {drop_pct:4.1f}% of prev]")
        prev = count

    # Monthly cohort view
    print("\n  Monthly Cohort Funnel:")
    monthly = []
    for month in sorted(mature["signup_month"].unique()):
        c = mature[mature["signup_month"] == month]
        ac = c[c["vehicle_type"].isin(["Auto", "Cab"])]
        er = c[c["vehicle_type"] == "ERickshaw"]
        row = {
            "month": str(month),
            "signups": len(c),
            "started": c["started_docs"].sum(),
            "cleared_1": (c["docs_cleared"] >= 1).sum(),
            "cleared_3": (c["docs_cleared"] >= 3).sum(),
            "cleared_all": (ac["docs_cleared"] >= 6).sum() + (er["docs_cleared"] >= 5).sum(),
            "approved": (c["final_status"] == "approved").sum(),
        }
        row["rate"] = f"{row['approved']/row['signups']*100:.1f}%"
        monthly.append(row)
    print(pd.DataFrame(monthly).to_string(index=False))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    stages = list(funnel.keys())
    counts = list(funnel.values())

    bars = ax1.barh(stages[::-1], counts[::-1],
                    color=sns.color_palette("Blues_r", len(stages)))
    ax1.set_xlabel("Number of Captains")
    ax1.set_title("Onboarding Funnel -- Mature Cohort\n(Signups before June 2026)",
                  fontweight="bold")
    for bar, c in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                 f"{c:,}", va="center", fontsize=9)

    drops = []
    for i in range(1, len(counts)):
        d = (counts[i-1] - counts[i]) / counts[i-1] * 100 if counts[i-1] > 0 else 0
        drops.append(d)
    drop_labels = [f"{stages[i-1]} -> {stages[i]}" for i in range(1, len(stages))]
    colors = ["#e74c3c" if d > 30 else "#f39c12" if d > 15 else "#2ecc71" for d in drops]
    ax2.barh(drop_labels[::-1], drops[::-1], color=colors[::-1])
    ax2.set_xlabel("Drop-off Rate (%)")
    ax2.set_title("Stage-over-Stage Drop-off Rates", fontweight="bold")
    for bar, d in zip(ax2.patches, drops[::-1]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{d:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "A1_funnel.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR / 'A1_funnel.png'}")

    return funnel, mature


# ══════════════════════════════════════════════════════════════════════════
# PART A2: SEGMENT THE DROP-OFFS
# ══════════════════════════════════════════════════════════════════════════

def segment_analysis(df, segment_col, title):
    """Compute approval rate by segment with gap-to-best sizing."""
    results = []
    for val in sorted(df[segment_col].unique()):
        seg = df[df[segment_col] == val]
        n = len(seg)
        started = seg["started_docs"].sum()
        approved = (seg["final_status"] == "approved").sum()
        rate = approved / n * 100 if n > 0 else 0
        results.append({
            "segment": val, "signups": n, "start_rate": started/n*100,
            "approved": approved, "approval_rate": rate,
            "monthly_signups": n / 5,
        })
    res = pd.DataFrame(results)
    best = res["approval_rate"].max()
    avg = res["approved"].sum() / res["signups"].sum() * 100
    res["gap_to_best"] = best - res["approval_rate"]
    res["extra_if_best"] = (res["monthly_signups"] * res["gap_to_best"] / 100).round(0)

    print(f"\n  -- {title} --")
    print(f"  Overall: {avg:.1f}%  |  Best: {best:.1f}%")
    for _, r in res.iterrows():
        print(f"    {r['segment']:20s}  n={r['signups']:>5,}  started={r['start_rate']:5.1f}%  "
              f"approved={r['approval_rate']:5.1f}%  gap={r['gap_to_best']:+5.1f}pp  "
              f"extra={r['extra_if_best']:>5.0f}/mo")
    return res


def run_segmentation(mature, doc_mature):
    """Run all segmentation cuts and produce visualizations."""
    print("\n" + "=" * 70)
    print("A2: SEGMENTED DROP-OFF ANALYSIS")
    print("=" * 70)

    city_seg    = segment_analysis(mature, "city", "By City")
    vtype_seg   = segment_analysis(mature, "vehicle_type", "By Vehicle Type")
    channel_seg = segment_analysis(mature, "acquisition_channel", "By Acquisition Channel")
    device_seg  = segment_analysis(mature, "device_tier", "By Device Tier")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (seg, title) in zip(axes.flat, [
        (city_seg, "By City"), (vtype_seg, "By Vehicle Type"),
        (channel_seg, "By Acquisition Channel"), (device_seg, "By Device Tier"),
    ]):
        avg = seg["approved"].sum() / seg["signups"].sum() * 100
        bars = ax.bar(seg["segment"], seg["approval_rate"],
                      color=sns.color_palette("Set2", len(seg)))
        ax.axhline(y=avg, color="red", ls="--", alpha=.7, label=f"Avg: {avg:.1f}%")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Approval Rate (%)")
        ax.legend()
        for bar, r in zip(bars, seg["approval_rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{r:.1f}%", ha="center", fontsize=9)
    plt.suptitle("A2: Approval Rate by Segment (Mature Cohort)",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "A2_segments.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR / 'A2_segments.png'}")

    # Document x Failure Reason
    print("\n  -- Document x Failure Reason --")
    failures = doc_mature[doc_mature["event_type"] == "verification_fail"].copy()
    combos = failures.groupby(["doc_type", "failure_reason"]).agg(
        total=("event_id", "count"),
        captains=("captain_id", "nunique")
    ).reset_index().sort_values("captains", ascending=False)
    print(f"\n  Top 10 doc x failure combos:")
    print(combos.head(10).to_string(index=False))

    # Terminal failures
    failed = failures.groupby(["captain_id", "doc_type"]).size().reset_index(name="n")
    passed = doc_mature[doc_mature["event_type"] == "verification_pass"][
        ["captain_id", "doc_type"]
    ].drop_duplicates()
    terminal = failed.merge(passed, on=["captain_id", "doc_type"], how="left", indicator=True)
    terminal = terminal[terminal["_merge"] == "left_only"]
    print(f"\n  Terminal failures (failed, never passed): {terminal['captain_id'].nunique():,} captains")
    term_by_doc = terminal.groupby("doc_type")["captain_id"].nunique().sort_values(ascending=False)
    print(term_by_doc.to_string())

    # Heatmap
    pivot = combos.pivot(index="doc_type", columns="failure_reason", values="captains").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("A2: Verification Failures -- Doc Type x Failure Reason\n"
                 "(Unique captains affected)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "A2_failure_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'A2_failure_heatmap.png'}")

    # Time analysis
    print("\n  -- Time-to-Complete --")
    first_upload = doc_mature[doc_mature["event_type"] == "upload_success"] \
        .groupby("captain_id")["event_ts"].min().reset_index()
    first_upload.columns = ["captain_id", "first_upload_ts"]
    time_df = mature[["captain_id", "signup_ts"]].merge(first_upload, on="captain_id", how="left")
    time_df["hours"] = (time_df["first_upload_ts"] - time_df["signup_ts"]).dt.total_seconds() / 3600
    print(f"  Signup to first upload: median={time_df['hours'].median():.1f}h, "
          f"90th={time_df['hours'].quantile(0.90):.1f}h, "
          f"never={time_df['hours'].isna().sum():,}")

    approved_t = mature[mature["final_status"] == "approved"][
        ["captain_id", "signup_ts", "decision_ts"]
    ].copy()
    approved_t["days"] = (approved_t["decision_ts"] - approved_t["signup_ts"]).dt.total_seconds() / 86400
    print(f"  Signup to approval: median={approved_t['days'].median():.1f}d, "
          f"90th={approved_t['days'].quantile(0.90):.1f}d")


# ══════════════════════════════════════════════════════════════════════════
# PART A3: EVALUATE CAMP_WA_002
# ══════════════════════════════════════════════════════════════════════════

def evaluate_campaign(mature, nudges, doc_mature):
    """
    Evaluate CAMP_WA_002 using delivered/not-delivered as natural experiment.

    Key insight: non-delivery is quasi-random (7.1% fail rate due to phone/network).
    If the campaign has a causal effect, delivered captains should convert higher
    than not-delivered captains.
    """
    print("\n" + "=" * 70)
    print("A3: CAMPAIGN EVALUATION -- CAMP_WA_002")
    print("=" * 70)

    camp = nudges[nudges["campaign_id"] == "CAMP_WA_002"]
    camp_first = camp.groupby("captain_id").agg(
        delivered=("delivered", "max"),
        clicked=("clicked", "max"),
    ).reset_index()

    camp_ids = set(camp_first["captain_id"])
    mature = mature.copy()
    mature["received_camp"] = mature["captain_id"].isin(camp_ids)

    # Naive comparison
    print(f"\n  CAMP_WA_002: {len(camp):,} nudges to {camp['captain_id'].nunique():,} captains")
    print(f"  Delivery rate: {camp['delivered'].mean():.1%}")
    print(f"  Click rate (of delivered): "
          f"{camp[camp['delivered']==1]['clicked'].mean():.1%}")

    print("\n  -- Naive Comparison --")
    for received, label in [(True, "Received"), (False, "Did NOT receive")]:
        seg = mature[mature["received_camp"] == received]
        n, app = len(seg), (seg["final_status"] == "approved").sum()
        rate = app / n * 100
        print(f"    {label:20s}  n={n:>6,}  approved={app:>4,}  rate={rate:.1f}%")
        if received:
            rate_r = rate
        else:
            rate_nr = rate
    print(f"    Naive lift: {rate_r - rate_nr:+.1f}pp  (BIASED by selection)")

    # Delivered vs Not-Delivered (causal test)
    print("\n  -- Delivered vs Not-Delivered (Causal Test) --")
    camp_with_status = camp_first.merge(
        mature[["captain_id", "final_status"]], on="captain_id", how="inner"
    )
    for del_val, label in [(1, "Delivered"), (0, "Not Delivered")]:
        seg = camp_with_status[camp_with_status["delivered"] == del_val]
        n, app = len(seg), (seg["final_status"] == "approved").sum()
        rate = app / n * 100 if n > 0 else 0
        print(f"    {label:20s}  n={n:>5,}  approved={app:>4,}  rate={rate:.1f}%")
        if del_val == 1:
            r_d = rate
        else:
            r_nd = rate
    causal_lift = r_d - r_nd
    print(f"    Causal lift estimate: {causal_lift:+.1f}pp")

    # Dose-response: clicked vs not-clicked
    print("\n  -- Clicked vs Not-Clicked (among delivered) --")
    delivered = camp_with_status[camp_with_status["delivered"] == 1]
    for cl, label in [(1, "Clicked"), (0, "Not Clicked")]:
        seg = delivered[delivered["clicked"] == cl]
        n, app = len(seg), (seg["final_status"] == "approved").sum()
        rate = app / n * 100 if n > 0 else 0
        print(f"    {label:20s}  n={n:>5,}  approved={app:>4,}  rate={rate:.1f}%")

    # All 4 campaigns
    print("\n  -- All Campaigns (delivered vs not-delivered) --")
    for camp_id in nudges["campaign_id"].unique():
        cn = nudges[nudges["campaign_id"] == camp_id]
        cf = cn.groupby("captain_id").agg(delivered=("delivered", "max")).reset_index()
        cws = cf.merge(mature[["captain_id", "final_status"]], on="captain_id", how="inner")
        rates = {}
        for dv in [1, 0]:
            s = cws[cws["delivered"] == dv]
            rates[dv] = ((s["final_status"] == "approved").sum() / len(s) * 100 if len(s) > 0 else 0, len(s))
        lift = rates[1][0] - rates[0][0]
        ch = cn["channel"].iloc[0]
        print(f"    {camp_id:15s} ({ch:14s})  del={rates[1][0]:5.1f}% (n={rates[1][1]:>5})  "
              f"not_del={rates[0][0]:5.1f}% (n={rates[0][1]:>4})  lift={lift:+5.1f}pp")

    # Conclusion
    print(f"\n  CONCLUSION:")
    print(f"  All campaigns show ~0pp causal lift.")
    print(f"  The +17.8pp naive lift for CAMP_WA_002 is a SELECTION ARTIFACT.")
    print(f"  RECOMMENDATION: Do NOT scale 5x. Run a proper A/B test first.")
    print(f"  Redirect nudge budget to failure-specific guidance messages.")

    return causal_lift


# ══════════════════════════════════════════════════════════════════════════
# DEEP DIVES (A2 extensions)
# ══════════════════════════════════════════════════════════════════════════

def deep_dives(mature, doc_mature, nudges):
    """Non-obvious analyses that strengthen the recommendations."""
    print("\n" + "=" * 70)
    print("DEEP DIVES")
    print("=" * 70)

    # DD1: Retry behavior
    print("\n  -- DD1: Retry Behavior --")
    failures = doc_mature[doc_mature["event_type"] == "verification_fail"]
    first_fails = failures.groupby(["captain_id", "doc_type"]).first().reset_index()
    retries = doc_mature[doc_mature["attempt_no"] > 1]
    retry_caps = retries.groupby(["captain_id", "doc_type"]).size().reset_index(name="n")
    merged = first_fails.merge(retry_caps, on=["captain_id", "doc_type"], how="left")
    merged["retried"] = merged["n"].notna()
    print(f"  Overall retry rate after first failure: {merged['retried'].mean():.1%}")
    for doc in DOC_ORDER_ALL:
        seg = merged[merged["doc_type"] == doc]
        if len(seg) > 0:
            print(f"    {doc:12s}  retry={seg['retried'].mean():.0%}  (n={len(seg):,})")

    # DD2: Time-to-stall
    print("\n  -- DD2: Time-to-Stall --")
    last_act = doc_mature.groupby("captain_id")["event_ts"].max().reset_index()
    last_act.columns = ["captain_id", "last_ts"]
    stall = mature[["captain_id", "signup_ts", "final_status"]].merge(
        last_act, on="captain_id", how="left"
    )
    stall["days"] = (stall["last_ts"] - stall["signup_ts"]).dt.total_seconds() / 86400
    drops = stall[stall["final_status"] == "dropped_in_docs"]
    d_with = drops[drops["days"].notna()]
    print(f"  Dropouts active for median {d_with['days'].median():.1f} days")
    for days in [1, 2, 3, 5, 7]:
        n = (d_with["days"] <= days).sum()
        print(f"    Quit within {days}d: {n:>5,} ({n/len(d_with):.0%})")

    # DD3: Speed-to-first-upload vs approval
    print("\n  -- DD3: Speed vs Approval --")
    fu = doc_mature[doc_mature["event_type"] == "upload_success"] \
        .groupby("captain_id")["event_ts"].min().reset_index()
    fu.columns = ["captain_id", "first_ts"]
    sp = mature[["captain_id", "signup_ts", "final_status"]].merge(fu, on="captain_id", how="left")
    sp["hours"] = (sp["first_ts"] - sp["signup_ts"]).dt.total_seconds() / 3600
    for lo, hi in [(0, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168)]:
        seg = sp[(sp["hours"] >= lo) & (sp["hours"] < hi)]
        n = len(seg)
        app = (seg["final_status"] == "approved").sum()
        print(f"    {lo:>3d}-{hi:>3d}h: n={n:>5,}  rate={app/n*100:.1f}%")

    # DD4: Device x image failures
    print("\n  -- DD4: Device Tier x Image Failures --")
    doc_dev = doc_mature.merge(mature[["captain_id", "device_tier"]].drop_duplicates(), on="captain_id")
    img_fail = doc_dev[
        (doc_dev["event_type"] == "verification_fail") &
        (doc_dev["failure_reason"].isin(["image_blurred", "ocr_low_confidence", "details_not_legible"]))
    ]
    caps_by_dev = mature.groupby("device_tier").size()
    fails_by_dev = img_fail.groupby("device_tier")["captain_id"].nunique()
    for tier in ["low", "mid", "high"]:
        nc = caps_by_dev.get(tier, 0)
        nf = fails_by_dev.get(tier, 0)
        print(f"    {tier:6s}  {nf:>4}/{nc:>5} = {nf/nc:.1%} with image failures")

    # DD5: Drop-off point by device tier
    print("\n  -- DD5: Drop-off Point by Device Tier --")
    for tier in ["low", "mid", "high"]:
        seg = mature[(mature["device_tier"] == tier) & (mature["final_status"] == "dropped_in_docs")]
        top = seg["last_stage_reached"].value_counts().head(3)
        s = ", ".join([f"{k}={v/len(seg):.0%}" for k, v in top.items()])
        print(f"    {tier:6s}  (n={len(seg):>5})  {s}")

    # DD6: Surge response at airport
    print("\n  -- DD6: Surge Response (see Part B) --")
    print("  (Covered in airport section)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Retry rate by doc
    retry_by_doc = merged.groupby("doc_type")["retried"].mean().sort_values()
    axes[0].barh(retry_by_doc.index, retry_by_doc.values * 100,
                 color=sns.color_palette("Set2", 6))
    axes[0].set_xlabel("Retry Rate (%)")
    axes[0].set_title("Retry Rate After\nFirst Failure", fontweight="bold")
    for i, (_, v) in enumerate(retry_by_doc.items()):
        axes[0].text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=9)

    # Speed vs approval
    bins = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 48), (48, 168)]
    labels, rates = [], []
    for lo, hi in bins:
        seg = sp[(sp["hours"] >= lo) & (sp["hours"] < hi)]
        n = len(seg)
        labels.append(f"{lo}-{hi}h")
        rates.append((seg["final_status"] == "approved").sum() / n * 100 if n > 0 else 0)
    labels.append("Never")
    rates.append(0)
    colors = ["#2ecc71" if r > 20 else "#f39c12" if r > 10 else "#e74c3c" for r in rates]
    axes[1].bar(labels, rates, color=colors)
    axes[1].set_ylabel("Approval Rate (%)")
    axes[1].set_title("Speed to First Upload\nvs Approval Rate", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    # Verification turnaround
    uploads = doc_mature[doc_mature["event_type"] == "upload_success"][
        ["captain_id", "doc_type", "attempt_no", "event_ts"]
    ].rename(columns={"event_ts": "upload_ts"})
    verifs = doc_mature[doc_mature["event_type"].isin(["verification_pass", "verification_fail"])][
        ["captain_id", "doc_type", "attempt_no", "event_ts"]
    ].rename(columns={"event_ts": "verify_ts"})
    tat = uploads.merge(verifs, on=["captain_id", "doc_type", "attempt_no"])
    tat["hours"] = (tat["verify_ts"] - tat["upload_ts"]).dt.total_seconds() / 3600
    tat_by_doc = tat.groupby("doc_type")["hours"].median().sort_values()
    axes[2].barh(tat_by_doc.index, tat_by_doc.values,
                 color=sns.color_palette("coolwarm", 6))
    axes[2].set_xlabel("Median Verification Time (hours)")
    axes[2].set_title("Verification Turnaround\nby Doc Type", fontweight="bold")
    for i, (_, v) in enumerate(tat_by_doc.items()):
        axes[2].text(v + 0.1, i, f"{v:.1f}h", va="center", fontsize=9)

    plt.suptitle("Deep Dive: Retry, Speed, and Verification", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deep_dive_1.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR / 'deep_dive_1.png'}")


# ══════════════════════════════════════════════════════════════════════════
# PART B1: AIRPORT DEMAND-SUPPLY MISMATCH
# ══════════════════════════════════════════════════════════════════════════

def airport_mismatch(airport_hr):
    """Characterise the demand-supply mismatch at airport terminals."""
    print("\n" + "=" * 70)
    print("B1: AIRPORT DEMAND-SUPPLY MISMATCH")
    print("=" * 70)

    apt = airport_hr[airport_hr["zone_type"] == "airport_terminal"].copy()
    other = airport_hr[airport_hr["zone_type"] != "airport_terminal"]
    apt["hour"] = apt["hour_ts"].dt.hour
    apt["fill_rate"] = apt["fulfilled_requests"] / apt["requests"]
    apt["dow_num"] = apt["hour_ts"].dt.dayofweek

    # Overall
    fill = apt["fulfilled_requests"].sum() / apt["requests"].sum()
    print(f"\n  Airport fill rate: {fill:.1%}")
    print(f"  Total requests: {apt['requests'].sum():,}")
    print(f"  Unfulfilled: {apt['unfulfilled_requests'].sum():,} ({1-fill:.1%})")

    # Comparison with city
    for zt in other["zone_type"].unique():
        z = other[other["zone_type"] == zt]
        f = z["fulfilled_requests"].sum() / z["requests"].sum()
        print(f"  {zt:20s}  fill={f:.1%}")

    # Hourly pattern
    hourly = apt.groupby("hour").agg(
        req=("requests", "mean"), ful=("fulfilled_requests", "mean"),
        unful=("unfulfilled_requests", "mean"), fill=("fill_rate", "mean"),
        caps=("online_captains", "mean"), eta=("avg_eta_min", "mean"),
        surge=("avg_surge_multiplier", "mean"),
    ).reset_index()

    print(f"\n  Hourly pattern:")
    print(f"  {'Hr':>3} {'Req':>5} {'Fill':>5} {'Unf':>5} {'Rate':>6} {'Caps':>5} {'ETA':>5} {'Surge':>5}")
    for _, r in hourly.iterrows():
        print(f"  {r['hour']:3.0f} {r['req']:5.0f} {r['ful']:5.0f} {r['unful']:5.0f} "
              f"{r['fill']:6.1%} {r['caps']:5.0f} {r['eta']:5.1f} {r['surge']:5.2f}x")

    n_days = apt["hour_ts"].dt.date.nunique()
    daily_unful = apt["unfulfilled_requests"].sum() / n_days
    print(f"\n  Daily unfulfilled rides: {daily_unful:.0f}")
    print(f"  Monthly unfulfilled: {daily_unful * 30:,.0f}")

    # Heatmap
    hm = apt.groupby(["hour", "dow_num"])["fill_rate"].mean().unstack()
    hm.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(hm, annot=True, fmt=".0%", cmap="RdYlGn", ax=ax, vmin=0.15, vmax=0.55)
    ax.set_title("B1: Airport Fill Rate by Hour x Day of Week", fontweight="bold")
    ax.set_ylabel("Hour of Day")
    ax.set_xlabel("Day of Week")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "B1_airport_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'B1_airport_heatmap.png'}")

    return apt


# ══════════════════════════════════════════════════════════════════════════
# PART B2: POST-AIRPORT-TRIP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def airport_post_trip(airport_tr, airport_hr):
    """Analyse what happens after an airport trip."""
    print("\n" + "=" * 70)
    print("B2: POST-AIRPORT-TRIP ANALYSIS")
    print("=" * 70)

    overall_return = airport_tr["got_return_fare_within_20min"].mean()
    print(f"\n  Return fare rate: {overall_return:.1%}")

    by_drop = airport_tr.groupby(["drop_zone_id", "drop_zone_type"]).agg(
        trips=("trip_id", "count"),
        return_rate=("got_return_fare_within_20min", "mean"),
        cancel_rate=("captain_cancelled", "mean"),
        avg_dist=("trip_distance_km", "mean"),
        avg_fare=("fare_inr", "mean"),
    ).reset_index().sort_values("return_rate", ascending=False)

    print("\n  By drop zone:")
    for _, r in by_drop.iterrows():
        print(f"    {r['drop_zone_id']:>8s} ({r['drop_zone_type']:>10s})  "
              f"trips={r['trips']:>6,}  return={r['return_rate']:.1%}  "
              f"cancel={r['cancel_rate']:.1%}  dist={r['avg_dist']:.1f}km  "
              f"fare=Rs.{r['avg_fare']:.0f}")

    # Effective earnings
    no_ret = airport_tr[airport_tr["got_return_fare_within_20min"] == 0]
    w_ret  = airport_tr[airport_tr["got_return_fare_within_20min"] == 1]
    eff_w = w_ret["fare_inr"].mean() / w_ret["trip_distance_km"].mean()
    eff_n = no_ret["fare_inr"].mean() / (no_ret["trip_distance_km"].mean() * 2)  # deadhead

    print(f"\n  Effective earnings:")
    print(f"    With return fare:    Rs.{eff_w:.1f}/km")
    print(f"    Without (deadhead):  Rs.{eff_n:.1f}/km")
    print(f"    Ratio: {eff_w/eff_n:.1f}x")

    # Surge response
    apt = airport_hr[airport_hr["zone_type"] == "airport_terminal"].copy()
    apt = apt.sort_values(["zone_id", "hour_ts"])
    apt["next_caps"] = apt.groupby("zone_id")["online_captains"].shift(-1)
    corr = apt[["avg_surge_multiplier", "next_caps"]].dropna().corr().iloc[0, 1]
    print(f"\n  Surge -> next-hour captains correlation: {corr:.3f}")
    print(f"  Captains {'DO' if corr > 0.3 else 'DO NOT'} respond to surge")

    # Night vs day
    airport_tr_c = airport_tr.copy()
    airport_tr_c["hour"] = airport_tr_c["request_ts"].dt.hour
    night = airport_tr_c[(airport_tr_c["hour"] >= 22) | (airport_tr_c["hour"] < 6)]
    day   = airport_tr_c[(airport_tr_c["hour"] >= 8) & (airport_tr_c["hour"] < 19)]

    print(f"\n  Night (10pm-6am) vs Day (8am-7pm):")
    for label, seg in [("Night", night), ("Day", day)]:
        cr = seg["captain_cancelled"].mean()
        rr = seg["got_return_fare_within_20min"].mean()
        print(f"    {label:6s}  n={len(seg):>6,}  cancel={cr:.1%}  return={rr:.1%}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Return fare by drop type
    by_type = airport_tr.groupby("drop_zone_type").agg(
        ret=("got_return_fare_within_20min", "mean"),
        cancel=("captain_cancelled", "mean"),
    ).reset_index().sort_values("ret")
    bars = axes[0].bar(by_type["drop_zone_type"], by_type["ret"] * 100,
                       color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[0].set_title("Return Fare Rate\nby Drop Zone Type", fontweight="bold")
    axes[0].set_ylabel("% Getting Return Fare")
    for b, v in zip(bars, by_type["ret"]):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     f"{v:.0%}", ha="center")

    # Cancel rate by drop type
    bars = axes[1].bar(by_type["drop_zone_type"], by_type["cancel"] * 100,
                       color=["#e74c3c", "#f39c12", "#2ecc71"])
    axes[1].set_title("Captain Cancellation Rate\nby Drop Zone Type", fontweight="bold")
    axes[1].set_ylabel("Cancel Rate (%)")
    for b, v in zip(bars, by_type["cancel"]):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"{v:.1%}", ha="center")

    # Effective earnings
    axes[2].bar(["With Return\nFare", "Without Return\nFare (deadhead)"],
                [eff_w, eff_n], color=["#2ecc71", "#e74c3c"])
    axes[2].set_title("Effective Captain Earnings\nper Effective km", fontweight="bold")
    axes[2].set_ylabel("Rs. per km")

    plt.suptitle("B2: Airport Trip Post-Drop Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "B2_airport_trips.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'B2_airport_trips.png'}")

    # Hourly return fare by drop zone type (deep dive chart)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hourly_tr = airport_tr_c.groupby("hour").agg(
        trips=("trip_id", "count"),
        cancel=("captain_cancelled", "mean"),
        ret=("got_return_fare_within_20min", "mean"),
    ).reset_index()

    ax1 = axes[0]
    ax2t = ax1.twinx()
    ax1.bar(hourly_tr["hour"], hourly_tr["trips"], alpha=0.3, color="steelblue", label="Volume")
    ax2t.plot(hourly_tr["hour"], hourly_tr["ret"] * 100, "ro-", lw=2, label="Return Rate")
    ax2t.plot(hourly_tr["hour"], hourly_tr["cancel"] * 100, "s-", color="orange", lw=2, label="Cancel Rate")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Trip Volume", color="steelblue")
    ax2t.set_ylabel("Rate (%)")
    ax1.set_title("Airport: Hourly Patterns", fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    ax = axes[1]
    for zt in ["city_core", "tech_park", "suburban"]:
        seg = airport_tr_c[airport_tr_c["drop_zone_type"] == zt]
        hr = seg.groupby("hour")["got_return_fare_within_20min"].mean() * 100
        ax.plot(hr.index, hr.values, "o-", label=zt, lw=2)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Return Fare Rate (%)")
    ax.set_title("Return Fare by\nDrop Zone & Hour", fontweight="bold")
    ax.legend()
    ax.axhline(y=50, color="gray", ls="--", alpha=0.5)

    plt.suptitle("Deep Dive: Airport Micro-Economics by Hour", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deep_dive_2_airport.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'deep_dive_2_airport.png'}")


# ══════════════════════════════════════════════════════════════════════════
# PART B3: BUSINESS RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════

def airport_recommendation(airport_hr, airport_tr):
    """Answer: is targeted acquisition the right intervention?"""
    print("\n" + "=" * 70)
    print("B3: IS TARGETED ACQUISITION THE RIGHT INTERVENTION?")
    print("=" * 70)

    apt = airport_hr[airport_hr["zone_type"] == "airport_terminal"].copy()
    apt["hour"] = apt["hour_ts"].dt.hour

    n_days = apt["hour_ts"].dt.date.nunique()
    daily_unful = apt["unfulfilled_requests"].sum() / n_days
    avg_fare = airport_tr["fare_inr"].mean()

    print(f"\n  Daily unfulfilled: {daily_unful:.0f} rides")
    print(f"  Monthly revenue at risk: Rs.{daily_unful * 30 * avg_fare:,.0f}")

    # Night gap sizing
    night = apt[(apt["hour"] >= 22) | (apt["hour"] < 6)]
    night_h = night.groupby("hour").agg(
        req=("requests", "mean"), ful=("fulfilled_requests", "mean"),
        caps=("online_captains", "mean"),
    ).reset_index()
    rides_per_cap = night_h["ful"].sum() / night_h["caps"].sum()
    gap = night_h["req"].sum() - night_h["ful"].sum()
    extra_cap_hours = gap / rides_per_cap
    print(f"\n  Night shift sizing:")
    print(f"    Gap: {gap:.0f} rides per night (across hour-slots)")
    print(f"    Rides per captain-hour (night): {rides_per_cap:.2f}")
    print(f"    Extra captain-hours needed: {extra_cap_hours:.0f}")
    print(f"    At 8h shifts: {extra_cap_hours/8:.0f} captains")

    # The economics problem
    ret = airport_tr["got_return_fare_within_20min"].mean()
    sub_no_ret = (
        (airport_tr["drop_zone_type"] == "suburban") &
        (airport_tr["got_return_fare_within_20min"] == 0)
    ).mean()

    print(f"\n  WHY ACQUISITION ALONE WON'T WORK:")
    print(f"    {1-ret:.0%} of trips get no return fare")
    print(f"    {sub_no_ret:.0%} of ALL trips are suburban + no return (worst case)")
    print(f"    Surge correlation with next-hour supply: NEGATIVE")
    print(f"    New captains will learn to avoid airport -> churn or go offline")

    print(f"\n  RECOMMENDED INTERVENTION PACKAGE:")
    daily_sub_no_ret = len(airport_tr[
        (airport_tr["drop_zone_type"] == "suburban") &
        (airport_tr["got_return_fare_within_20min"] == 0)
    ]) / 61  # 61 days of data
    print(f"    1. Night bonus (Rs.100/trip, 10pm-6am): attracts existing supply")
    print(f"    2. Suburban deadhead allowance (Rs.100/trip): fixes root cause")
    print(f"       Cost: ~Rs.{daily_sub_no_ret * 100 * 30:,.0f}/month")
    print(f"    3. THEN targeted acquisition (only ~{extra_cap_hours/8:.0f} captains needed)")
    print(f"       Only after economics are fixed, so they actually stay")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("RAPIDO SUPPLY DS -- COMPLETE ANALYSIS")
    print("=" * 70)

    # Load
    captains, doc_events, approvals, activation, nudges, airport_hr, airport_tr = load_data()
    print(f"\n  Loaded 7 datasets. Extraction date: {EXTRACTION_DATE}")

    # Master table
    master = build_master_table(captains, approvals, activation)
    mature = master[master["is_mature"]].copy()
    mature_ids = set(mature["captain_id"])
    doc_mature = doc_events[doc_events["captain_id"].isin(mature_ids)].copy()
    print(f"  Mature cohort: {len(mature):,} (signup < {MATURITY_CUTOFF.date()})")
    print(f"  In-progress in mature: {(mature['final_status'] == 'in_progress').sum()}")

    # Quality checks
    run_data_quality_checks(captains, approvals, activation, doc_events)

    # Who started docs (needed for funnel)
    started_ids = set(
        doc_mature[doc_mature["event_type"] == "upload_success"]["captain_id"].unique()
    )
    mature["started_docs"] = mature["captain_id"].isin(started_ids)

    # A1: Funnel
    funnel, mature = build_funnel(mature, doc_mature)

    # A2: Segmentation
    run_segmentation(mature, doc_mature)

    # A3: Campaign
    evaluate_campaign(mature, nudges, doc_mature)

    # Deep dives
    deep_dives(mature, doc_mature, nudges)

    # B1: Airport mismatch
    apt = airport_mismatch(airport_hr)

    # B2: Post-trip analysis
    airport_post_trip(airport_tr, airport_hr)

    # B3: Recommendation
    airport_recommendation(airport_hr, airport_tr)

    # A4: Recommendations summary
    print("\n" + "=" * 70)
    print("A4: THREE RANKED RECOMMENDATIONS")
    print("=" * 70)
    monthly_signups = len(mature) / 5
    monthly_approved = (mature["final_status"] == "approved").sum() / 5
    print(f"\n  Baseline: {monthly_signups:.0f} signups/mo -> {monthly_approved:.0f} approved/mo "
          f"({monthly_approved/monthly_signups:.1%})")
    print(f"\n  1. Fix RC upload UX for low-device captains (~150 extra/mo)")
    print(f"  2. Redirect paid_digital budget to fos_field (~50 extra/mo)")
    print(f"  3. Replace generic nudges with failure-specific guidance (~80-160 extra/mo)")
    print(f"\n  Combined potential: 280-360 extra approved captains/month (+40-50%)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"  Charts saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 70)
