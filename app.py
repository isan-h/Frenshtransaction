"""
app.py
------
Test the trained models on a NEW description you type in yourself --
text only, no other input -- compare what all models predict, and log
corrections for later retraining.

Run with:
    pip install streamlit
    python -m streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import predict_category, list_available_models, get_categories  # noqa: E402
from feedback import save_correction, pending_count  # noqa: E402

st.set_page_config(page_title="Effyis Category Classifier", page_icon="\U0001F3F7\uFE0F")

st.title("\U0001F3F7\uFE0F Effyis Category Classifier")
st.caption("Type a transaction description (only) and compare what each model predicts.")

available = list_available_models()
if not available:
    st.error("No trained models found in models/. Run `python src/evaluate.py` first.")
    st.stop()

description = st.text_input("Description", "CB LIDL 4658 CERGY")
feedback_model_label = st.selectbox(
    "Model to give feedback on", list(available.keys()), index=0,
    help="All models below get predicted either way -- this just picks which one your correction applies to.",
)

if st.button("Predict with ALL models", type="primary"):
    rows = []
    for label, model_key in available.items():
        full, primary, detailed, confidence, all_probs = predict_category(description, model_key)
        rows.append({
            "Model": label,
            "Predicted category": full,
            "Primary": primary,
            "Detailed": detailed,
            "Confidence": f"{confidence*100:.1f}%" if confidence is not None else "n/a",
        })

    st.session_state["last_predictions"] = rows
    st.session_state["last_description"] = description
    st.session_state["last_feedback_model_label"] = feedback_model_label
    st.session_state["feedback_saved"] = False

rows = st.session_state.get("last_predictions")
if rows:
    st.subheader("Predictions")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    agree = len(set(r["Predicted category"] for r in rows)) == 1
    if agree:
        st.success(f"All models agree: **{rows[0]['Predicted category']}**")
    else:
        st.warning("Models disagree on this one -- worth a closer look.")

    st.divider()
    fb_label = st.session_state["last_feedback_model_label"]
    fb_row = next(r for r in rows if r["Model"] == fb_label)
    st.write(f"**Feedback on {fb_label}'s prediction:** {fb_row['Predicted category']}")

    categories = get_categories()
    default_index = (
        categories.index(fb_row["Predicted category"])
        if fb_row["Predicted category"] in categories
        else 0
    )
    corrected = st.selectbox("Correct category", categories, index=default_index, key="corrected_category")

    if st.button("Save feedback"):
        save_correction(
            description=st.session_state["last_description"],
            model_used=available[fb_label],
            predicted_category=fb_row["Predicted category"],
            correct_category=corrected,
        )
        st.session_state["feedback_saved"] = True

    if st.session_state.get("feedback_saved"):
        if corrected == fb_row["Predicted category"]:
            st.success("Logged as confirmed correct. Thanks!")
        else:
            st.success(f"Logged correction: '{fb_row['Predicted category']}' -> '{corrected}'.")

    n_pending = pending_count()
    if n_pending > 0:
        st.caption(
            f"\U0001F4DD {n_pending} correction(s) saved and waiting to be folded into training data. "
            "Run `python src/incorporate_feedback.py` then `python src/evaluate.py` to retrain on them "
            "(or just use run_app.py, which does this automatically when you stop the app)."
        )

st.divider()
st.caption(
    "For the full model-by-model accuracy comparison across all 2,220 "
    "transactions, see data/processed/predictions_comparison.csv -- built "
    "from the real unlabeled file, with honest (cross_val_predict) match rates."
)