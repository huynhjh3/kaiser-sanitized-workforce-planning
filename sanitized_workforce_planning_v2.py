"""
Sanitized Workforce Planning Analytics — Portfolio Version V2

This script mirrors the public-safe methodology of a confidential workforce
planning project using mock data only.

It demonstrates:
1. Intake lifecycle filtering
2. Timestamp sequence correction using median stage-gap imputation
3. TAT feature engineering
4. Primary employee attribution
5. Team-level demand forecasting input
6. Capacity gap and utilization calculation
7. Operational delivery effectiveness score
8. Core-flex staffing recommendation logic
9. Mock dashboard output generation

No client records, internal team names, employee data, or confidential outputs are included.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("../data")
OUT_DIR = Path("../dashboard")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. Load mock data
# -----------------------------
intake = pd.read_csv(DATA_DIR / "mock_intake_records.csv", parse_dates=[
    "received_on", "specialist_assigned_on", "actual_start", "delivered_on", "actual_end"
])
employees = pd.read_csv(DATA_DIR / "mock_employee_lookup.csv")
capacity = pd.read_csv(DATA_DIR / "mock_team_capacity.csv")

# -----------------------------
# 2. Filter completed lifecycle records
# -----------------------------
completed = intake[intake["status"].eq("Completed")].copy()

# -----------------------------
# 3. Timestamp correction
# Median stage-gap imputation:
# received_on → specialist_assigned_on → actual_start → delivered_on → actual_end
# -----------------------------
stages = ["received_on", "specialist_assigned_on", "actual_start", "delivered_on", "actual_end"]

def median_gap_days(df, start_col, end_col):
    valid = df[[start_col, end_col]].dropna()
    valid = valid[valid[end_col] >= valid[start_col]]
    if valid.empty:
        return 1.0
    return (valid[end_col] - valid[start_col]).dt.total_seconds().median() / 86400

stage_gaps = {}
for a, b in zip(stages[:-1], stages[1:]):
    stage_gaps[(a, b)] = median_gap_days(completed, a, b)

def correct_timestamps(row):
    # Forward impute missing/out-of-order timestamp with previous timestamp + median gap
    for a, b in zip(stages[:-1], stages[1:]):
        if pd.isna(row[b]) or (pd.notna(row[a]) and row[b] < row[a]):
            if pd.notna(row[a]):
                row[b] = row[a] + pd.Timedelta(days=stage_gaps[(a, b)])
    return row

completed = completed.apply(correct_timestamps, axis=1)

# -----------------------------
# 4. TAT feature engineering
# -----------------------------
completed["total_tat_days"] = (completed["actual_end"] - completed["received_on"]).dt.total_seconds() / 86400
completed["assignment_tat_days"] = (completed["specialist_assigned_on"] - completed["received_on"]).dt.total_seconds() / 86400
completed["pre_work_tat_days"] = (completed["actual_start"] - completed["specialist_assigned_on"]).dt.total_seconds() / 86400
completed["execution_tat_days"] = (completed["delivered_on"] - completed["actual_start"]).dt.total_seconds() / 86400
completed["closure_tat_days"] = (completed["actual_end"] - completed["delivered_on"]).dt.total_seconds() / 86400

# -----------------------------
# 5. Primary employee attribution
# Default: assigned_to
# Fallback: assigned_specialist → tester_completed_by
# -----------------------------
completed["primary_employee"] = completed["assigned_to"]
completed["primary_employee"] = completed["primary_employee"].fillna(completed["assigned_specialist"])
completed["primary_employee"] = completed["primary_employee"].fillna(completed["tester_completed_by"])

completed = completed.merge(
    employees,
    left_on="primary_employee",
    right_on="employee_id",
    how="left"
)

# -----------------------------
# 6. Operational Delivery Effectiveness Score
# Public-safe version:
# Effectiveness = (Complexity Weight × Avg Concurrent Requests) ÷ Relative TAT
# -----------------------------

effort_map = {"Low": 1, "Medium": 2, "High": 3}
completed["effort_score"] = completed["effort_level"].map(effort_map).fillna(1)
completed["project_score"] = completed["project_flag"].astype(int) * 4

team_median_tat = completed.groupby("analysis_team")["execution_tat_days"].median().rename("team_median_tat")
completed = completed.merge(team_median_tat, on="analysis_team", how="left")

# Higher team TAT percentile = higher complexity signal
team_tat_rank = completed.groupby("analysis_team")["execution_tat_days"].median().rank(pct=True)
team_tat_score = (team_tat_rank * 3).rename("team_tat_score").reset_index()
completed = completed.merge(team_tat_score, on="analysis_team", how="left")

completed["complexity_weight_raw"] = completed["project_score"] + completed["effort_score"] + completed["team_tat_score"]
# normalize roughly to 1-10
min_raw = completed["complexity_weight_raw"].min()
max_raw = completed["complexity_weight_raw"].max()
completed["complexity_weight"] = 1 + 9 * (completed["complexity_weight_raw"] - min_raw) / (max_raw - min_raw)

# Mock concurrent requests as count of requests by employee-month
completed["month"] = completed["received_on"].dt.to_period("M").astype(str)
employee_month_counts = completed.groupby(["primary_employee", "month"]).size().rename("monthly_concurrent_requests").reset_index()
avg_concurrency = employee_month_counts.groupby("primary_employee")["monthly_concurrent_requests"].mean().rename("avg_concurrent_requests").reset_index()
completed = completed.merge(avg_concurrency, on="primary_employee", how="left")

employee_median_tat = completed.groupby("primary_employee")["execution_tat_days"].median().rename("employee_median_tat").reset_index()
completed = completed.merge(employee_median_tat, on="primary_employee", how="left")
completed["relative_tat"] = completed["employee_median_tat"] / completed["team_median_tat"]
completed["relative_tat"] = completed["relative_tat"].replace([np.inf, -np.inf], np.nan).fillna(1)

completed["effectiveness_score"] = (
    completed["complexity_weight"] * completed["avg_concurrent_requests"] / completed["relative_tat"]
)

team_effectiveness = completed.groupby("analysis_team")["effectiveness_score"].median().reset_index()
team_effectiveness["effectiveness_score"] = team_effectiveness["effectiveness_score"].round(2)

# -----------------------------
# 7. Capacity analysis
# Estimated Monthly Capacity = Median Productivity × Stable Contributors
# Capacity Gap = Forecasted Demand − Estimated Capacity
# Utilization Ratio = Forecast Demand ÷ Estimated Capacity
# -----------------------------
capacity["estimated_monthly_capacity"] = (
    capacity["median_productivity_per_contributor"] * capacity["stable_contributors"]
)
capacity["capacity_gap"] = (
    capacity["forecast_monthly_demand_2026"] - capacity["estimated_monthly_capacity"]
)
capacity["utilization_ratio"] = (
    capacity["forecast_monthly_demand_2026"] / capacity["estimated_monthly_capacity"]
)

def pressure_status(utilization):
    if utilization < 0.9:
        return "Excess Capacity"
    if utilization <= 1.1:
        return "Balanced"
    if utilization <= 1.3:
        return "Moderate Pressure"
    return "High/Critical Pressure"

capacity["sustainability_risk"] = capacity["utilization_ratio"].apply(pressure_status)
capacity = capacity.merge(team_effectiveness, on="analysis_team", how="left")

# -----------------------------
# 8. Core-flex staffing recommendation logic
# -----------------------------
def recommend(row):
    if row["utilization_ratio"] > 1.3 and row["effectiveness_score"] >= capacity["effectiveness_score"].median():
        return "Layer 1: Stable Core / Dedicated Support"
    if row["utilization_ratio"] > 1.1:
        return "Layer 2: Flexible Delivery"
    if row["utilization_ratio"] < 0.9:
        return "Layer 3: Floating Pool Source"
    return "Monitor Quarterly"

capacity["core_flex_recommendation"] = capacity.apply(recommend, axis=1)

# -----------------------------
# 9. Export dashboard-ready tables and visuals
# -----------------------------
completed.to_csv(OUT_DIR / "cleaned_mock_intake_with_tat.csv", index=False)
capacity.to_csv(OUT_DIR / "capacity_recommendation_summary.csv", index=False)

# Chart 1: Utilization
plot_df = capacity.sort_values("utilization_ratio", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(plot_df["analysis_team"], plot_df["utilization_ratio"])
plt.axvline(1.0, linestyle="--")
plt.title("Mock Capacity Utilization Ratio by Team")
plt.xlabel("Forecast Demand / Estimated Capacity")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_utilization_ratio.png", dpi=150)
plt.close()

# Chart 2: Capacity gap
plt.figure(figsize=(10, 6))
plt.bar(capacity["analysis_team"], capacity["capacity_gap"])
plt.axhline(0, linestyle="--")
plt.title("Mock Monthly Capacity Gap by Team")
plt.ylabel("Forecasted Demand - Estimated Capacity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_capacity_gap.png", dpi=150)
plt.close()

# Chart 3: TAT stage medians
tat_cols = ["assignment_tat_days", "pre_work_tat_days", "execution_tat_days", "closure_tat_days"]
tat_summary = completed[tat_cols].median().reset_index()
tat_summary.columns = ["tat_stage", "median_days"]

plt.figure(figsize=(9, 5))
plt.bar(tat_summary["tat_stage"], tat_summary["median_days"])
plt.title("Mock Median TAT by Lifecycle Stage")
plt.ylabel("Median Days")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_tat_stage_medians.png", dpi=150)
plt.close()

# Chart 4: Effectiveness
eff_plot = capacity.sort_values("effectiveness_score")
plt.figure(figsize=(10, 6))
plt.barh(eff_plot["analysis_team"], eff_plot["effectiveness_score"])
plt.title("Mock Operational Delivery Effectiveness Score")
plt.xlabel("Effectiveness Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_effectiveness_score.png", dpi=150)
plt.close()

print("Sanitized V2 analysis complete.")
print(capacity[[
    "analysis_team", "forecast_monthly_demand_2026", "estimated_monthly_capacity",
    "utilization_ratio", "capacity_gap", "sustainability_risk",
    "effectiveness_score", "core_flex_recommendation"
]])
