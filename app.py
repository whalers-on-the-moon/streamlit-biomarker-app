import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Inspiration4 Biomarker Profile Viewer", layout="wide"
)

# CSS layout optimization for 1080p display viewing without vertical scrolling
st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        h1 { font-size: 1.6rem !important; padding-bottom: 0.5rem !important; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Inspiration4 Astronaut Longitudinal Biomarker Profiles")

dataset_options = {
    "Comprehensive Metabolic Panel (CMP)": (
        "LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv"
    ),
    "Multiplex Serum Immune (Eve Panel)": (
        "LSDS-8_Multiplex_serum.immune.EvePanel_SUBMITTED.csv"
    ),
}

selected_label = st.sidebar.selectbox(
    "Select Dataset", list(dataset_options.keys())
)
file_path = dataset_options[selected_label]


@st.cache_data
def load_data(path):
  return pd.read_csv(path)


if os.path.exists(file_path):
  df_raw = load_data(file_path)
  cols = list(df_raw.columns)

  # 1. Target / Analyte Column Detection
  ana_c = next(
      (
          c
          for c in cols
          if any(
              k in c.lower()
              for k in [
                  "analyte",
                  "biomarker",
                  "target",
                  "factor",
                  "cytokine",
                  "assay",
                  "parameter",
                  "test",
              ]
          )
      ),
      None,
  )

  # 2. Subject / Astronaut ID Column Detection
  sub_candidates = [c for c in cols if c != ana_c]
  sub_c = next(
      (
          c
          for c in sub_candidates
          if c.lower().strip()
          in [
              "subject_id",
              "subject",
              "subject id",
              "id",
              "sample_id",
              "sample id",
              "participant",
              "participant_id",
              "astronaut",
          ]
      ),
      None,
  )
  if not sub_c:
    sub_c = next(
        (
            c
            for c in sub_candidates
            if any(
                k in c.lower()
                for k in ["subject", "participant", "patient", "donor", "astronaut"]
            )
            or c.lower().strip() == "id"
        ),
        sub_candidates[0],
    )

  # 3. Timepoint Column Detection
  tp_candidates = [c for c in sub_candidates if c != sub_c]
  tp_c = next(
      (
          c
          for c in tp_candidates
          if any(
              k in c.lower()
              for k in ["time", "point", "tp", "visit", "stage", "phase"]
          )
      ),
      tp_candidates[0],
  )

  # 4. Value / Concentration Column Detection
  val_candidates = [c for c in tp_candidates if c != tp_c]
  val_c = next(
      (
          c
          for c in val_candidates
          if any(
              k in c.lower()
              for k in [
                  "val",
                  "conc",
                  "res",
                  "result",
                  "pg",
                  "ng",
                  "mg",
                  "amount",
                  "level",
                  "reading",
              ]
          )
      ),
      val_candidates[-1] if val_candidates else cols[-1],
  )

  df = pd.DataFrame({
      "Subject_ID": df_raw[sub_c].astype(str).str.strip(),
      "Timepoint": df_raw[tp_c].astype(str).str.strip(),
      "Value": df_raw[val_c],
      "Analyte": (
          df_raw[ana_c].astype(str).str.strip() if ana_c else "Biomarker"
      ),
  })

  analytes = sorted(df["Analyte"].dropna().unique())
  default_idx = next(
      (i for i, a in enumerate(analytes) if a.upper() == "GLUCOSE"), 0
  )
  target_analyte = st.sidebar.selectbox(
      "Select Biomarker / Analyte", analytes, index=default_idx
  )
  df_filtered = df[df["Analyte"] == target_analyte].copy()

  df_filtered["Value"] = pd.to_numeric(
      df_filtered["Value"].astype(str).str.replace(r"[^\d.-]", "", regex=True),
      errors="coerce",
  )

  timepoint_days = {
      "L-92": -92,
      "L-44": -44,
      "L-3": -3,
      "R+1": 4,
      "R+45": 48,
      "R+82": 85,
  }
  df_filtered["Days"] = df_filtered["Timepoint"].str.upper().map(timepoint_days)

  df_clean = df_filtered.dropna(subset=["Subject_ID", "Days", "Value"]).copy()

  if not df_clean.empty:
    df_clean["Days"] = df_clean["Days"].astype(int)

    summary_df = (
        df_clean.groupby("Days")["Value"]
        .agg(
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(hspace=0.08)

    # Layer 0: Continuous Mission Phase Shading & Event Boundaries
    for ax in (ax1, ax2):
      ax.axvspan(-100, 0, color="#e8f0fe", alpha=0.6, zorder=0)
      ax.axvspan(0, 3, color="#fce8e6", alpha=0.9, zorder=0)
      ax.axvspan(3, 90, color="#e6f4ea", alpha=0.6, zorder=0)
      ax.axvline(0, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=1)
      ax.axvline(3, color="#1976d2", linestyle="--", linewidth=1.5, zorder=1)

    # Layer 1 & 2: Trajectories & Group Summary Ribbon
    ax1.fill_between(
        summary_df["Days"],
        summary_df["q25"],
        summary_df["q75"],
        color="#bdbdbd",
        alpha=0.4,
        zorder=2,
    )
    ax1.plot(
        summary_df["Days"],
        summary_df["median"],
        color="#212121",
        linewidth=2.5,
        zorder=3,
    )

    subjects = sorted(df_clean["Subject_ID"].unique())
    palette = sns.color_palette("Set2", len(subjects))
    for idx, sub in enumerate(subjects):
      sub_data = df_clean[df_clean["Subject_ID"] == sub].sort_values("Days")
      ax1.plot(
          sub_data["Days"],
          sub_data["Value"],
          marker="o",
          linewidth=1.5,
          color=palette[idx],
          zorder=4,
      )

    # Layer 3: Distribution Boxplots
    unique_days = sorted(df_clean["Days"].unique())
    for day in unique_days:
      vals = df_clean[df_clean["Days"] == day]["Value"]
      ax2.boxplot(
          vals,
          positions=[day],
          widths=6,
          patch_artist=True,
          boxprops=dict(facecolor="#e0e0e0", zorder=2),
          medianprops=dict(color="black", zorder=3),
      )

    tp_labels = {v: k for k, v in timepoint_days.items()}
    xtick_labels = [
        f"{tp_labels.get(d, '')}\n({int(round(d))}d)" for d in unique_days
    ]

    ax2.set_xticks(unique_days)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlim(-100, 90)

    ax1.set_ylabel(f"{target_analyte} Level", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel(
        "Mission Timepoint (Days Relative to Launch)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_title(
        f"Inspiration4 Astronauts: Longitudinal {target_analyte} Profile",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax1.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax2.grid(True, linestyle=":", alpha=0.6, zorder=0)

    # Render Legend in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Chart Legend")

    st.sidebar.markdown("**Phases & Events**")
    st.sidebar.markdown(
        "🟦 Pre-Flight (-100d to 0d)<br>🟥 In-Flight (0d to 3d)<br>🟩 Post-Flight"
        " (3d to 90d)",
        unsafe_allow_html=True,
    )
   
    st.sidebar.markdown("**Astronaut Trajectories**")
    for idx, sub in enumerate(subjects):
      rgb = palette[idx]
      hex_color = "#%02x%02x%02x" % (
          int(rgb[0] * 255),
          int(rgb[1] * 255),
          int(rgb[2] * 255),
      )
      st.sidebar.markdown(
          f"<span style='color:{hex_color}; font-weight:bold;'>● Subject"
          f" {sub}</span>",
          unsafe_allow_html=True,
      )

    st.pyplot(fig, use_container_width=True)
else:
  st.error(f"Dataset non-existent: {file_path}")
