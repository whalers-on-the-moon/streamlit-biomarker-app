import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Inspiration4 Biomarker Profile Viewer", layout="wide")
st.title("Inspiration4 Astronaut Longitudinal Biomarker Profiles")

dataset_options = {
    "Comprehensive Metabolic Panel (CMP)": "LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv",
    "Multiplex Serum Immune (Eve Panel)": "LSDS-8_Multiplex_serum.immune.EvePanel_SUBMITTED.csv",
}

selected_label = st.sidebar.selectbox("Select Dataset", list(dataset_options.keys()))
file_path = dataset_options[selected_label]

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if os.path.exists(file_path):
    df_raw = load_data(file_path)
    col_map = {}
    for col in df_raw.columns:
        c = col.strip().lower()
        if "subject" in c or "sample" in c:
            col_map[col] = "Subject_ID"
        elif "time" in c or "point" in c:
            col_map[col] = "Timepoint"
        elif "analyte" in c:
            col_map[col] = "Analyte"
        elif any(k in c for k in ["val", "conc", "res", "result"]):
            col_map[col] = "Value"

    df = df_raw.rename(columns=col_map).loc[:, lambda x: ~x.columns.duplicated()].copy()

    if "Analyte" in df.columns:
        analytes = sorted(df["Analyte"].dropna().astype(str).str.strip().unique())
        default_idx = next((i for i, a in enumerate(analytes) if a.upper() == "GLUCOSE"), 0)
        target_analyte = st.sidebar.selectbox("Select Biomarker / Analyte", analytes, index=default_idx)
        df_filtered = df[df["Analyte"] == target_analyte].copy()
    else:
        df_filtered = df.copy()
        target_analyte = "Analyte"

    df_filtered["Value"] = pd.to_numeric(df_filtered["Value"], errors="coerce")
    timepoint_days = {"L-92": -92, "L-44": -44, "L-3": -3, "R+1": 4, "R+45": 48, "R+82": 85}
    df_filtered["Days"] = df_filtered["Timepoint"].astype(str).str.strip().map(timepoint_days)
    df_clean = df_filtered.dropna(subset=["Subject_ID", "Days", "Value"]).copy()

    if not df_clean.empty:
        summary_df = df_clean.groupby("Days")["Value"].agg(
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75)
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        fig.subplots_adjust(hspace=0.08)

        for ax in (ax1, ax2):
            ax.axvspan(-100, 0, color="#e8f0fe", alpha=0.6, zorder=0, label="Pre-Flight")
            ax.axvspan(0, 3, color="#fce8e6", alpha=0.9, zorder=0, label="In-Flight (3d)")
            ax.axvspan(3, 90, color="#e6f4ea", alpha=0.6, zorder=0, label="Post-Flight")
            ax.axvline(0, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=1, label="Launch (L=0)")
            ax.axvline(3, color="#1976d2", linestyle="--", linewidth=1.5, zorder=1, label="Return (R=0)")

        ax1.fill_between(summary_df["Days"], summary_df["q25"], summary_df["q75"], color="#bdbdbd", alpha=0.4, zorder=2, label="Group IQR")
        ax1.plot(summary_df["Days"], summary_df["median"], color="#212121", linewidth=2.5, zorder=3, label="Group Median")

        subjects = sorted(df_clean["Subject_ID"].unique())
        palette = sns.color_palette("Set2", len(subjects))
        for idx, sub in enumerate(subjects):
            sub_data = df_clean[df_clean["Subject_ID"] == sub].sort_values("Days")
            ax1.plot(sub_data["Days"], sub_data["Value"], marker="o", linewidth=1.5, color=palette[idx], zorder=4, label=str(sub))

        unique_days = sorted(df_clean["Days"].unique())
        for day in unique_days:
            vals = df_clean[df_clean["Days"] == day]["Value"]
            ax2.boxplot(vals, positions=[day], widths=6, patch_artist=True, boxprops=dict(facecolor="#e0e0e0", zorder=2), medianprops=dict(color="black", zorder=3))

        tp_labels = {v: k for k, v in timepoint_days.items()}
        ax2.set_xticks(unique_days)
        ax2.set_xticklabels([f"{tp_labels.get(d, '')}\n({d}d)" for d in unique_days])
        ax2.set_xlim(-100, 90)

        ax1.set_ylabel(f"{target_analyte} Level", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mission Timepoint (Days Relative to Launch)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Inspiration4 Astronauts: Longitudinal {target_analyte} Profile", fontsize=14, fontweight="bold", pad=12)

        ax1.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax2.grid(True, linestyle=":", alpha=0.6, zorder=0)

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9, ncols=2)

        st.pyplot(fig)
else:
    st.error(f"Dataset non-existent: {file_path}")
