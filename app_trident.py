import os
import json
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sklearn


OUTPUT_DIR = "outputs"
RESULT_FILE = os.path.join(OUTPUT_DIR, "results_testkl.csv")
ABLATION_FILE = os.path.join(OUTPUT_DIR, "ablation_results.csv")
MODE_FILES = {
    "zero-shot": os.path.join(OUTPUT_DIR, "results_testkl_zero_shot.csv"),
    "few-shot": os.path.join(OUTPUT_DIR, "results_testkl_few_shot.csv"),
    "cot": os.path.join(OUTPUT_DIR, "results_testkl_cot.csv"),
}
BIN_LABELS = ["suitable", "unsuitable"]

st.set_page_config(page_title="HYDROGPT‑TRIDENT Dashboard", layout="wide")

st.title("HYDROGPT‑TRIDENT — Groundwater Quality Dashboard")


def evaluate_hallucination(df, pred_col):
    valid_labels = set(df["WQC_true"].dropna().unique())
    predicted = df[pred_col].dropna()

    if len(predicted) == 0:
        return 0, 0.0

    invalid_mask = ~predicted.isin(valid_labels)
    hallucination_count = int(invalid_mask.sum())
    hallucination_rate = hallucination_count / len(predicted)
    return hallucination_count, hallucination_rate


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def get_index_column(df):
    if "index" in df.columns:
        return "index"
    return None


def get_labels(df):
    return BIN_LABELS


def parse_claude_json(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def cap_display_value(value):
    if pd.isna(value):
        return value
    return min(float(value), 0.99999999)


def load_mode_results():
    mode_results = {}
    for mode, path in MODE_FILES.items():
        if os.path.exists(path):
            mode_results[mode] = pd.read_csv(path)
    return mode_results


def summarize_mode_df(df):
    summary = {
        "GPT_Mode": df["gpt_mode"].iloc[0] if "gpt_mode" in df.columns and len(df) else "unknown",
        "Accuracy": cap_display_value(accuracy_score(df["WQC_true"], df["Final_label"])),
        "Macro_F1": cap_display_value(f1_score(df["WQC_true"], df["Final_label"], average="macro")),
        "GPT_Accuracy": cap_display_value(accuracy_score(df["WQC_true"], df["GPT_label"])),
        "GPT_Macro_F1": cap_display_value(f1_score(df["WQC_true"], df["GPT_label"], average="macro")),
        "Full_Hallucinations": evaluate_hallucination(df, "Final_label")[0],
        "GPT_Hallucinations": evaluate_hallucination(df, "GPT_label")[0],
        "Claude_Agreement": cap_display_value(df["Claude_confidence"].mean()) if "Claude_confidence" in df.columns else float("nan"),
    }
    return summary

if not os.path.exists(RESULT_FILE):
    st.error(f"Results file not found: {RESULT_FILE}")
else:
    df = pd.read_csv(RESULT_FILE)
    index_col = get_index_column(df)
    labels = get_labels(df)
    gpt_mode = df["gpt_mode"].iloc[0] if "gpt_mode" in df.columns and len(df) else "unknown"
    mode_results = load_mode_results()

    full_acc = cap_display_value(accuracy_score(df["WQC_true"], df["Final_label"]))
    full_f1 = cap_display_value(f1_score(df["WQC_true"], df["Final_label"], average="macro"))
    full_hallu_count, full_hallu_rate = evaluate_hallucination(df, "Final_label")

    who_acc = cap_display_value(accuracy_score(df["WQC_true"], df["WQC_true"]))
    who_f1 = cap_display_value(f1_score(df["WQC_true"], df["WQC_true"], average="macro"))
    who_hallu_count, who_hallu_rate = evaluate_hallucination(df, "WQC_true")

    gpt_acc = cap_display_value(accuracy_score(df["WQC_true"], df["GPT_label"]))
    gpt_f1 = cap_display_value(f1_score(df["WQC_true"], df["GPT_label"], average="macro"))
    gpt_hallu_count, gpt_hallu_rate = evaluate_hallucination(df, "GPT_label")

    claude_agreement = cap_display_value(df["Claude_confidence"].mean()) if "Claude_confidence" in df.columns else float("nan")

    ablation_df = None
    if os.path.exists(ABLATION_FILE):
        ablation_df = pd.read_csv(ABLATION_FILE, index_col=0)

    # Sidebar
    st.sidebar.header("Controls")
    if index_col is not None:
        sample_idx = st.sidebar.number_input(
            "Sample index",
            min_value=int(df[index_col].min()),
            max_value=int(df[index_col].max()),
            value=int(df[index_col].min()),
            step=1,
        )
    else:
        sample_idx = st.sidebar.number_input("Sample index", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)

    st.sidebar.caption(f"Rows loaded: {len(df)}")
    st.sidebar.caption(f"GPT mode: {gpt_mode}")

    overview, ablation, comparison, details, judge = st.tabs(["Overview", "Ablation", "Mode Comparison", "Sample Details", "Claude Judge"])

    with overview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("WHO Accuracy", f"{who_acc:.4f}")
            st.caption(f"Hallucinations: {who_hallu_count} | Rate: {who_hallu_rate:.4f}")
        with col2:
            st.metric("GPT Accuracy", f"{gpt_acc:.4f}")
            st.caption(f"Hallucinations: {gpt_hallu_count} | Rate: {gpt_hallu_rate:.4f}")
        with col3:
            st.metric("Full Accuracy", f"{full_acc:.4f}")
            st.caption(f"Hallucinations: {full_hallu_count} | Rate: {full_hallu_rate:.4f}")
        with col4:
            st.metric("Claude Agreement", f"{claude_agreement:.4f}" if pd.notna(claude_agreement) else "N/A")

        st.caption(f"GPT mode used for this run: {gpt_mode}")

        st.subheader("Full System Confusion Matrix")
        st.pyplot(plot_confusion_matrix(df["WQC_true"], df["Final_label"], labels, "Full System Confusion Matrix"))

        st.subheader("Misclassified Samples")
        error_cols = [col for col in [index_col, "WQC_true", "Final_label", "GPT_label", "Claude_confidence"] if col and col in df.columns]
        st.dataframe(df[df["WQC_true"] != df["Final_label"]][error_cols])

    with ablation:
        st.subheader("Ablation Comparison")
        if ablation_df is not None:
            st.dataframe(ablation_df)
        else:
            st.info(f"Ablation file not found: {ABLATION_FILE}")

        left, middle, right = st.columns(3)
        with left:
            st.caption("WHO-only confusion matrix")
            st.pyplot(plot_confusion_matrix(df["WQC_true"], df["WQC_true"], labels, "WHO-only Confusion Matrix"))
        with middle:
            st.caption("GPT-only confusion matrix")
            st.pyplot(plot_confusion_matrix(df["WQC_true"], df["GPT_label"], labels, "GPT-only Confusion Matrix"))
        with right:
            st.caption("Full-system confusion matrix")
            st.pyplot(plot_confusion_matrix(df["WQC_true"], df["Final_label"], labels, "Full System Confusion Matrix"))

    with comparison:
        st.subheader("Zero-shot / Few-shot / CoT Comparison")
        if mode_results:
            summary_rows = []
            for mode, mode_df in mode_results.items():
                summary = summarize_mode_df(mode_df)
                summary["Mode_File"] = os.path.basename(MODE_FILES[mode])
                summary_rows.append(summary)

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df)

            if len(summary_df) >= 2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(summary_df["GPT_Mode"], summary_df["Accuracy"], color="#4C72B0", alpha=0.8, label="Full Accuracy")
                ax.bar(summary_df["GPT_Mode"], summary_df["Macro_F1"], color="#55A868", alpha=0.7, label="Full Macro F1")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Score")
                ax.set_title("GPT Mode Comparison")
                ax.legend()
                st.pyplot(fig)
        else:
            st.info("No mode-specific result files found in outputs/")

    with details:
        st.subheader("Single Sample View")
        if index_col is not None:
            indexed = df[df[index_col] == sample_idx]
        else:
            indexed = df.iloc[[int(sample_idx)]]

        if indexed.empty:
            st.warning("No sample found for the selected index.")
        else:
            row = indexed.iloc[0]
            st.write("True WQC:", row["WQC_true"])
            st.write("Final Label:", row["Final_label"])
            st.write("GPT Label:", row["GPT_label"])
            if "Claude_confidence" in df.columns:
                st.write("Claude Confidence:", row["Claude_confidence"])
            if "Eval_comments" in df.columns:
                st.write("Eval Comments:", row["Eval_comments"])

    with judge:
        st.subheader("Claude as Judge")
        if index_col is not None:
            judge_row = df[df[index_col] == sample_idx]
        else:
            judge_row = df.iloc[[int(sample_idx)]]

        if judge_row.empty:
            st.warning("No sample found for the selected index.")
        else:
            row = judge_row.iloc[0]
            claude_json = parse_claude_json(row["Claude_raw_json"]) if "Claude_raw_json" in df.columns else None

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Judge Confidence", f"{cap_display_value(row['Claude_confidence']):.4f}" if "Claude_confidence" in df.columns else "N/A")
            with c2:
                st.metric("Low Confidence", str(bool(row["Low_confidence"])) if "Low_confidence" in df.columns else "N/A")
            with c3:
                st.metric("Valid JSON", str(bool(claude_json.get("is_valid_json"))) if claude_json else "N/A")
            with c4:
                st.metric("Label Allowed", str(bool(claude_json.get("label_is_allowed"))) if claude_json else "N/A")

            st.write("True WQC:", row.get("WQC_true", "N/A"))
            st.write("GPT Label:", row.get("GPT_label", "N/A"))
            st.write("Final Label:", row.get("Final_label", "N/A"))
            st.write("Claude Comments:", row.get("Eval_comments", "N/A"))

            if claude_json:
                st.json(claude_json)
            else:
                st.info("Claude raw JSON is not available in this results file.")
