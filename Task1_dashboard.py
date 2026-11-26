# Run: streamlit run Task1_dashboard.py

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional RDKit import for 2D molecule rendering (graceful fallback if unavailable)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Molecular Solubility Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.html("""
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 2rem !important;
}
</style>
""")

st.title("🧪 Molecular Solubility Dashboard")
st.caption(
    "ESOL Dataset (Delaney, 2004) | Graph Neural Network Research | Task 1 & Question 1 Analysis"
)

# ==========================================
# PROJECT PATHS & DATA LOADING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "delaney-processed.csv"
RESULTS_DIR = BASE_DIR / "artifacts" / "results"


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_results(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


df = load_data()

baseline_df = load_results("baseline_comparison.csv")
architecture_df = load_results("gnn_architecture_comparison.csv")
pooling_df = load_results("graphsage_pooling_comparison.csv")
hidden_df = load_results("hidden_dimension_comparison.csv")
layer_df = load_results("layer_comparison.csv")
lr_df = load_results("learning_rate_comparison.csv")
dropout_df = load_results("dropout_comparison.csv")
split_comp_df = load_results("split_comparison.csv")
final_df = load_results("final_model_results.csv")
curves_df = load_results("final_training_curves.csv")
mol_preds_df = load_results("molecule_predictions.csv")

# ==========================================
# COLLAPSIBLE DROPDOWN: PROJECT OVERVIEW & FINAL MODEL METRICS
# ==========================================

with st.expander("📋 Project Overview & Current Best Model Performance", expanded=False):
    st.write(
        "This dashboard documents **Task 1 (Training & Model Comparison)** and **Question 1 (Model Performance)** "
        "for predicting measured water solubility ($\log S$ in mol/L) from molecular graph structure on the ESOL dataset."
    )

    total_col, train_col, val_col, test_col, seed_col = st.columns(5)
    total_col.metric("Total Molecules", f"{len(df):,}" if not df.empty else "1,128")
    train_col.metric("Training Split (70%)", "789")
    val_col.metric("Validation Split (15%)", "169")
    test_col.metric("Test Split (15%)", "170")
    seed_col.metric("Random Seed", "42")

    st.divider()
    st.markdown("##### Current Best Model Performance (GraphSAGE + Sum Pooling, 64 Hidden, 2 Layers)")

    if final_df is not None:
        result = final_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test RMSE (log S)", f"{result['Test RMSE']:.4f} log S")
        c2.metric("Test MAE (log S)", f"{result['Test MAE']:.4f} log S")
        c3.metric("Test R²", f"{result['Test R2']:.4f}")
        c4.metric("Within ±0.5 log S", f"{result['Accuracy ±0.5 (%)']:.2f}%")
        c5.metric("Within ±1.0 log S", f"{result['Accuracy ±1.0 (%)']:.2f}%")
    else:
        st.info("Run `model_training.ipynb` to generate `final_model_results.csv`.")
# ==========================================
# NAVIGATION TABS FOR TASK 1 & QUESTION 1
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Task 1: Model & Hyperparameter Comparisons",
    "🔀 Task 1: Data Split & Training Curves",
    "🔍 Question 1: Errors by Solubility, Structure & Split",
    "🧬 Question 1: Interactive Molecule Inspector & Findings"
])

# ==============================================================================
# TAB 1: MODEL ARCHITECTURE, POOLING & HYPERPARAMETERS (SLIDE 5)
# ==============================================================================
with tab1:
    st.subheader("Core Architecture, Pooling & Hidden Size")

    # Single shared dropdown controlling all 6 comparison charts in Tab 1
    ctrl_col, _ = st.columns([1, 2])
    with ctrl_col:
        selected_metric = st.selectbox(
            "Select Evaluation Metric (log S):",
            options=["Validation RMSE", "Validation MAE"],
            index=0,
            key="global_tab1_metric"
        )
    unit_suffix = " (log S)"

    col_a, col_b, col_c = st.columns(3)
    # Chart 1: Architecture Comparison
    with col_a:
        st.markdown("##### 1. GNN Architecture Comparison")
        if architecture_df is not None:
            fig1 = px.bar(
                architecture_df,
                x="Model",
                y=selected_metric,
                color="Model",
                text_auto=".4f",
                title="GCN vs GraphSAGE vs GAT",
                template="plotly_white"
            )
            fig1.update_layout(
                height=350,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="GNN Architecture"
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("`gnn_architecture_comparison.csv` unavailable.")

    # Chart 2: Pooling Comparison
    with col_b:
        st.markdown("##### 2. GraphSAGE Pooling Comparison")
        if pooling_df is not None:
            fig2 = px.bar(
                pooling_df,
                x="Pooling",
                y=selected_metric,
                color="Pooling",
                text_auto=".4f",
                title="Mean vs Max vs Sum Pooling",
                template="plotly_white"
            )
            fig2.update_layout(
                height=350,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="Pooling Method"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("`graphsage_pooling_comparison.csv` unavailable.")

    # Chart 3: Hidden Dimension Comparison
    with col_c:
        st.markdown("##### 3. Hidden Dimension Comparison")
        if hidden_df is not None:
            hidden_plot_df = hidden_df.copy()
            hidden_plot_df["Hidden Channels"] = hidden_plot_df["Hidden Channels"].astype(str)
            fig3 = px.bar(
                hidden_plot_df,
                x="Hidden Channels",
                y=selected_metric,
                color="Hidden Channels",
                text_auto=".4f",
                title="GraphSAGE: 64 vs 128 Channels",
                template="plotly_white"
            )
            fig3.update_layout(
                height=350,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="Hidden Channels"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("`hidden_dimension_comparison.csv` unavailable.")

    st.divider()
    st.subheader("Additional Hyperparameter Investigations (Layers, Learning Rate & Dropout)")
    col_d, col_e, col_f = st.columns(3)

    # Chart 4: Number of Layers
    with col_d:
        st.markdown("##### 4. Number of GNN Layers (2 vs 3 vs 4)")
        if layer_df is not None:
            layer_plot_df = layer_df.copy()
            layer_plot_df["Layers"] = layer_plot_df["Layers"].astype(str)
            fig_layer = px.bar(
                layer_plot_df,
                x="Layers",
                y=selected_metric,
                color="Layers",
                text_auto=".4f",
                title="Effect of Message-Passing Depth",
                template="plotly_white"
            )
            fig_layer.update_layout(
                height=340,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="Number of Layers"
            )
            st.plotly_chart(fig_layer, use_container_width=True)
        else:
            st.info("Run New Cell 2 in notebook to generate `layer_comparison.csv`.")

    # Chart 5: Learning Rate
    with col_e:
        st.markdown("##### 5. Learning Rate (0.0005 vs 0.001 vs 0.005)")
        if lr_df is not None:
            lr_plot_df = lr_df.copy()
            lr_plot_df["Learning Rate"] = lr_plot_df["Learning Rate"].astype(str)
            fig_lr = px.bar(
                lr_plot_df,
                x="Learning Rate",
                y=selected_metric,
                color="Learning Rate",
                text_auto=".4f",
                title="Effect of Adam Learning Rate",
                template="plotly_white"
            )
            fig_lr.update_layout(
                height=340,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="Learning Rate"
            )
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Run New Cell 3 in notebook to generate `learning_rate_comparison.csv`.")

    # Chart 6: Dropout Rate
    with col_f:
        st.markdown("##### 6. Dropout Regularization (0.0 vs 0.2 vs 0.4)")
        if dropout_df is not None:
            drop_plot_df = dropout_df.copy()
            drop_plot_df["Dropout"] = drop_plot_df["Dropout"].astype(str)
            fig_drop = px.bar(
                drop_plot_df,
                x="Dropout",
                y=selected_metric,
                color="Dropout",
                text_auto=".4f",
                title="Effect of Dropout Rate",
                template="plotly_white"
            )
            fig_drop.update_layout(
                height=340,
                showlegend=False,
                yaxis_title=f"{selected_metric}{unit_suffix}",
                xaxis_title="Dropout Probability"
            )
            st.plotly_chart(fig_drop, use_container_width=True)
        else:
            st.info("Run New Cell 4 in notebook to generate `dropout_comparison.csv`.")

    # Baseline comparison table
    if baseline_df is not None:
        with st.expander("📋 View Mean Baseline vs. Initial GCN Baseline Table"):
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: SPLIT INVESTIGATION & TRAINING CURVES (SLIDE 5)
# ==============================================================================
with tab2:
    col_s1, col_s2 = st.columns([1.15, 1])

    with col_s1:
        st.subheader("Separate Split Effects from Model Changes")
        st.caption(
            "Comparing GCN (Mean Pool) vs. GraphSAGE (Sum Pool) across multiple random seeds and split ratios."
        )
        if split_comp_df is not None:
            split_metric = st.selectbox(
                "Select Split Evaluation Metric",
                ["Validation RMSE (log S)", "Validation MAE (log S)", "Test RMSE (log S)", "Test MAE (log S)"],
                key="split_eval_metric"
            )
            fig_split = px.bar(
                split_comp_df,
                x="Split Setting",
                y=split_metric,
                color="Model",
                barmode="group",
                text_auto=".4f",
                title="Model Comparison Within Each Data Split",
                template="plotly_white"
            )
            fig_split.update_layout(height=380, yaxis_title=split_metric)
            st.plotly_chart(fig_split, use_container_width=True)
            st.dataframe(split_comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run New Cell 5 in notebook to generate `split_comparison.csv`.")

    with col_s2:
        st.subheader("Final Model Training & Validation Curves")
        st.caption("Recorded across 100 epochs (Seed = 42) for the selected GraphSAGE (Sum Pool) checkpoint.")
        if curves_df is not None:
            loss_unit = st.radio(
                "Loss Scale",
                ["RMSE (log S)", "MSE (log S²)"],
                horizontal=True
            )
            y_cols = ["Train_RMSE", "Validation_RMSE"] if "RMSE" in loss_unit else ["Train_MSE", "Validation_MSE"]
            best_row = curves_df.loc[curves_df["Validation_RMSE"].idxmin()]

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=curves_df["Epoch"],
                y=curves_df[y_cols[0]],
                mode="lines",
                name="Train"
            ))
            fig_curve.add_trace(go.Scatter(
                x=curves_df["Epoch"],
                y=curves_df[y_cols[1]],
                mode="lines",
                name="Validation"
            ))
            fig_curve.add_vline(
                x=int(best_row["Epoch"]),
                line_dash="dash",
                line_color="green",
                annotation_text=f"Best Epoch ({int(best_row['Epoch'])})"
            )
            fig_curve.update_layout(
                title=f"Training vs. Validation {loss_unit}",
                xaxis_title="Epoch",
                yaxis_title=loss_unit,
                template="plotly_white",
                height=420
            )
            st.plotly_chart(fig_curve, use_container_width=True)
        else:
            st.warning("`final_training_curves.csv` not found. Run Cell 36 in the notebook.")

# ==============================================================================
# TAB 3: QUESTION 1 — ERRORS BY SOLUBILITY, STRUCTURE & SPLIT (SLIDE 6)
# ==============================================================================
with tab3:
    st.subheader("Question 1: Where does the model perform well, and where does it fail?")

    if mol_preds_df is not None:
        # Filter controls
        selected_splits = st.multiselect(
            "Filter by Dataset Split:",
            options=["Train", "Validation", "Test"],
            default=["Train", "Validation", "Test"]
        )
        filtered_preds = mol_preds_df[mol_preds_df["Split"].isin(selected_splits)].copy()

        row1_c1, row1_c2 = st.columns(2)

        # 1. Parity Plot & Errors by Split
        with row1_c1:
            st.markdown("##### A. Predicted vs. Measured Solubility by Split")
            min_val = float(min(filtered_preds["Measured_logS"].min(), filtered_preds["Predicted_logS"].min())) - 0.5
            max_val = float(max(filtered_preds["Measured_logS"].max(), filtered_preds["Predicted_logS"].max())) + 0.5

            fig_parity = px.scatter(
                filtered_preds,
                x="Measured_logS",
                y="Predicted_logS",
                color="Split",
                hover_name="Compound ID",
                hover_data=["smiles", "Absolute_Error", "Molecular Weight"],
                opacity=0.75,
                title="Parity Plot: Measured vs. Predicted log S",
                template="plotly_white"
            )
            fig_parity.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="black"),
                name="Ideal (y = x)"
            ))
            fig_parity.update_layout(
                xaxis_title="Measured Solubility (log S)",
                yaxis_title="Predicted Solubility (log S)",
                height=400
            )
            st.plotly_chart(fig_parity, use_container_width=True)

        # 2. Error Distribution by Split (RMSE & MAE table + Boxplot)
        with row1_c2:
            st.markdown("##### B. Prediction Error Distribution Across Splits")
            fig_box = px.box(
                filtered_preds,
                x="Split",
                y="Absolute_Error",
                color="Split",
                points="outliers",
                hover_name="Compound ID",
                title="Absolute Error (|Pred - Measured|) by Split (log S)",
                template="plotly_white"
            )
            fig_box.update_layout(
                xaxis_title="Dataset Split",
                yaxis_title="Absolute Error (log S)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        row2_c1, row2_c2 = st.columns(2)

        # 3. Errors by Solubility Regime
        with row2_c1:
            st.markdown("##### C. Errors by Solubility Level (Measured log S)")
            filtered_preds["Solubility Bin"] = pd.cut(
                filtered_preds["Measured_logS"],
                bins=[-12, -6, -4, -2, 0, 2],
                labels=[
                    "Very Insoluble (<-6)",
                    "Low (-6 to -4)",
                    "Moderate (-4 to -2)",
                    "Soluble (-2 to 0)",
                    "Highly Soluble (>0)"
                ]
            )
            sol_bin_summary = (
                filtered_preds.groupby("Solubility Bin", observed=False)
                .agg(
                    Mean_Abs_Error=("Absolute_Error", "mean"),
                    RMSE=("Error (Pred - Actual)", lambda x: np.sqrt(np.mean(x**2))),
                    Count=("Compound ID", "count")
                )
                .reset_index()
            )
            fig_sol_err = px.bar(
                sol_bin_summary,
                x="Solubility Bin",
                y=["Mean_Abs_Error", "RMSE"],
                barmode="group",
                text_auto=".3f",
                hover_data=["Count"],
                title="RMSE & MAE Across Solubility Regimes (log S)",
                template="plotly_white"
            )
            fig_sol_err.update_layout(
                xaxis_title="Measured Solubility Regime (log S)",
                yaxis_title="Error (log S)",
                height=400
            )
            st.plotly_chart(fig_sol_err, use_container_width=True)

        # 4. Errors by Molecular Structure Feature
        with row2_c2:
            st.markdown("##### D. Errors by Molecular Structure Descriptor")
            struct_features = [
                "Molecular Weight",
                "Polar Surface Area",
                "Number of Rings",
                "Number of H-Bond Donors",
                "Number of Rotatable Bonds",
                "Minimum Degree"
            ]
            chosen_struct = st.selectbox(
                "Select Structural Descriptor:",
                struct_features,
                key="struct_feature_select"
            )
            fig_struct = px.scatter(
                filtered_preds,
                x=chosen_struct,
                y="Absolute_Error",
                color="Split",
                hover_name="Compound ID",
                hover_data=["Measured_logS", "Predicted_logS", "smiles"],
                trendline="ols" if chosen_struct in ["Molecular Weight", "Polar Surface Area"] else None,
                title=f"Absolute Error (log S) vs. {chosen_struct}",
                template="plotly_white"
            )
            fig_struct.update_layout(
                xaxis_title=chosen_struct,
                yaxis_title="Absolute Error (log S)",
                height=350
            )
            st.plotly_chart(fig_struct, use_container_width=True)

    else:
        st.warning("`molecule_predictions.csv` not found. Please run Cell 36 in `model_training.ipynb`.")

# ==============================================================================
# TAB 4: INTERACTIVE MOLECULE INSPECTOR & DOCUMENTED FINDINGS (SLIDE 6 & 7)
# ==============================================================================
with tab4:
    st.subheader("Connect Plots to Selected Molecules")

    if mol_preds_df is not None:
        insp_col1, insp_col2 = st.columns([1.2, 1])

        with insp_col1:
            filter_mode = st.radio(
                "Quick Molecule Filter:",
                [
                    "All Molecules",
                    "Top 20 Worst Predictions (Highest Error in Test Split)",
                    "Top 20 Best Predictions (Lowest Error in Test Split)",
                    "Slide Reference Molecules (Benzene, Phenol, Caffeine)"
                ],
                horizontal=False
            )

            if filter_mode == "Top 20 Worst Predictions (Highest Error in Test Split)":
                subset_df = (
                    mol_preds_df[mol_preds_df["Split"] == "Test"]
                    .sort_values("Absolute_Error", ascending=False)
                    .head(20)
                )
            elif filter_mode == "Top 20 Best Predictions (Lowest Error in Test Split)":
                subset_df = (
                    mol_preds_df[mol_preds_df["Split"] == "Test"]
                    .sort_values("Absolute_Error", ascending=True)
                    .head(20)
                )
            elif filter_mode == "Slide Reference Molecules (Benzene, Phenol, Caffeine)":
                subset_df = mol_preds_df[
                    mol_preds_df["Compound ID"].str.lower().isin(["benzene", "phenol", "caffeine"])
                ]
                if subset_df.empty:
                    subset_df = mol_preds_df
            else:
                subset_df = mol_preds_df.sort_values("Absolute_Error", ascending=False)

            subset_df = subset_df.copy()
            subset_df["Display_Label"] = (
                subset_df["Compound ID"].astype(str)
                + " | Split: " + subset_df["Split"].astype(str)
                + " | |Error|: " + subset_df["Absolute_Error"].round(3).astype(str) + " log S"
            )

            selected_label = st.selectbox(
                "Select a Molecule to Inspect:",
                subset_df["Display_Label"].tolist()
            )
            mol_row = subset_df[subset_df["Display_Label"] == selected_label].iloc[0]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Measured log S", f"{mol_row['Measured_logS']:.3f} log S")
            m2.metric("GNN Predicted log S", f"{mol_row['Predicted_logS']:.3f} log S")
            m3.metric("Signed Error", f"{mol_row['Error (Pred - Actual)']:+.3f} log S")
            m4.metric("ESOL Linear Pred", f"{mol_row['ESOL predicted log solubility in mols per litre']:.3f} log S")

            st.markdown(f"**Compound ID:** `{mol_row['Compound ID']}` | **Split:** `{mol_row['Split']}` | **Molecule Index:** `{mol_row['Molecule_Index']}`")
            st.markdown(f"**SMILES:** `{mol_row['smiles']}`")

            desc_df = pd.DataFrame([{
                "Molecular Weight": mol_row["Molecular Weight"],
                "Polar Surface Area": mol_row["Polar Surface Area"],
                "Rings": mol_row["Number of Rings"],
                "H-Bond Donors": mol_row["Number of H-Bond Donors"],
                "Rotatable Bonds": mol_row["Number of Rotatable Bonds"],
                "Min Degree": mol_row["Minimum Degree"]
            }])
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

        with insp_col2:
            st.markdown("##### 2D Chemical Structure (RDKit)")
            if RDKIT_AVAILABLE:
                mol_obj = Chem.MolFromSmiles(str(mol_row["smiles"]))
                if mol_obj is not None:
                    img = Draw.MolToImage(mol_obj, size=(380, 300))
                    st.image(img, caption=f"{mol_row['Compound ID']} ({mol_row['smiles']})")
                else:
                    st.warning("Could not parse SMILES with RDKit.")
            else:
                st.info("Install `rdkit` in your Streamlit environment to render 2D molecular structure images.")

        with st.expander("📄 View Full Molecule Predictions Table"):
            st.dataframe(
                mol_preds_df[[
                    "Molecule_Index", "Compound ID", "Split", "Measured_logS",
                    "Predicted_logS", "Error (Pred - Actual)", "Absolute_Error",
                    "Molecular Weight", "Number of Rings", "Polar Surface Area", "smiles"
                ]].sort_values("Absolute_Error", ascending=False),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

# ==========================================
    # DOCUMENTED DASHBOARD FINDINGS (SLIDE 6 & 7)
    # ==========================================
    st.subheader("📝 Documented Dashboard Findings (For Deadline 1 & Shared Model Meeting)")
    st.markdown("""
    * **1. Architecture & Pooling Selection (`GraphSAGE` + `sum` pooling):**
      * Across all tested GNN architectures (`GCN`, `GraphSAGE`, `GAT`), **GraphSAGE** achieved the lowest validation error (`1.1961 log S` vs. `1.3358 log S` for GAT and `1.3982 log S` for GCN).
      * Replacing `mean` or `max` pooling with **global `sum` pooling** produced the largest performance jump (Validation RMSE improved from `1.2248 log S` with mean pooling to **`0.8626 log S`** with sum pooling; $R^2 = 0.8523$). Chemically, water solubility depends on the *total count* of hydrophobic carbons and hydrophilic polar groups—which `sum` pooling preserves and `mean` pooling dilutes.
    * **2. Hyperparameter & Split Stability:**
      * **Hyperparameter Sensitivity:** `64` hidden channels (`0.8793 log S` vs. `0.9350 log S` for `128` channels) avoided overfitting on the 1,128-molecule ESOL dataset. Deeper message passing (`3–4` layers) and lower dropout (`0.0`) further reduced validation RMSE (`0.7872` and `0.7237 log S`), while our 2-layer baseline (`dropout = 0.2`, `lr = 0.001`) provides a well-regularized, over-smoothing-resistant checkpoint (`Test RMSE = 0.9559 log S`).
      * **Separating Split Effects from Model Changes:** Across all tested split seeds and ratios (`70/15/15 Seed 42`, `70/15/15 Seed 100`, `80/10/10 Seed 42`), **GraphSAGE (Sum Pool)** consistently outperformed **GCN (Mean Pool)** by ~0.35–0.50 $\log S$ RMSE, confirming the improvement comes from the model architecture rather than a lucky split.
    * **3. Question 1 — Where the Model Performs Well vs. Fails:**
      * **Performs Well:** Moderate-to-soluble small organic molecules ($\log S \in [-4, 0]$, Molecular Weight $< 250\text{ g/mol}$, 0–2 rings), where 77.65% of all test predictions fall within $\pm 1.0 \log S$.
      * **Where it Fails:** Extreme low-solubility compounds ($\log S < -6$) and large polycyclic/halogenated molecules (high Molecular Weight $> 350\text{ g/mol}$ or $\ge 4$ rings). Since our 5 node features only capture basic atom properties (`AtomicNum`, `Degree`, `FormalCharge`, `TotalNumHs`, `IsAromatic`) without 3D conformation or crystal-packing lattice energy, solid-state polycyclic molecules exhibit higher residual errors.
    * **4. Candidate Checkpoint Recommendation for XAI Explainer Comparison:**
      * We recommend `artifacts/checkpoints/final_graphsage_model.pt` (**2-layer GraphSAGE, Sum Pooling, 64 hidden channels, Test RMSE = `0.9559 log S`, Test MAE = `0.6727 log S`**) paired with `artifacts/results/split_ids.csv` (`Seed = 42`).
    """)

st.divider()
st.caption(
    "Task 1 & Question 1 — Molecular Water Solubility Prediction | "
    "Reproducible GNN Training & Error Analysis Dashboard"
)
