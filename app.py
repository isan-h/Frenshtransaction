import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Transaction Classification",
    layout="centered",
)

st.title(" Transaction Classification Client")
st.caption(
    "Client application communicating with the FastAPI prediction service."
)

description = st.text_input(
    "Transaction description",
    placeholder="CB LIDL 4658 CERGY",
)

if st.button("Predict", use_container_width=True):

    if description.strip() == "":
        st.warning("Enter a transaction description.")
        st.stop()

    try:

        response = requests.post(
            API_URL,
            json={
                "description": description
            },
            timeout=10,
        )

        if response.status_code != 200:
            st.error(response.text)
            st.stop()

        result = response.json()

        st.success("Prediction completed")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Category")

            st.metric(
                "Primary",
                result["primary_category"],
            )

            st.metric(
                "Detailed",
                result["detailed_category"],
            )

            if result["category_confidence"] is not None:
                st.progress(result["category_confidence"])
                st.write(
                    f"Confidence: {result['category_confidence']:.1%}"
                )

        with col2:

            st.subheader("Merchant")

            st.metric(
                "Merchant",
                result["merchant"],
            )

            if result["merchant_confidence"] is not None:
                st.progress(result["merchant_confidence"])
                st.write(
                    f"Confidence: {result['merchant_confidence']:.1%}"
                )

        st.divider()

        with st.expander("Raw API Response"):

            st.json(result)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the FastAPI server.\n\n"
            "Start it first with:\n\n"
            "uvicorn src.api:app --reload"
        )

    except Exception as e:

        st.exception(e)