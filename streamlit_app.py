"""
streamlit_app.py — ISR Risk Predictor (QFR/RWS)

Deployment copy. Model: random survival forest, 7 predictors selected by stability
selection over 300 nested bootstrap resamples; performance reported in the manuscript
comes from the nested validation in nested_bootstrap.py, not from this app.
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="ISR Risk Predictor", layout="wide")


@st.cache_resource
def load_model():
    import urllib.request

    path = os.path.join(os.path.dirname(__file__), "streamlit_model.joblib")
    if not os.path.exists(path):
        url = "https://github.com/Reinct/QFR/releases/latest/download/streamlit_model.joblib"
        st.info("Downloading model (~2 MB)...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # pragma: no cover
            st.error(f"Download failed: {e}")
            st.stop()
    return joblib.load(path)


st.title("ISR Risk Predictor — QFR/RWS (research demonstration)")
st.markdown("*In-stent restenosis prediction based on QFR and RWS*")

st.warning(
    "**For research and educational use only.** This application displays a model that was "
    "developed and internally validated in a single retrospective cohort (302 patients, 118 "
    "events); it has not been externally validated. Discrimination in nested internal "
    "validation was moderate (C-index about 0.66 to 0.70). It is not a medical device, has not "
    "been reviewed or cleared by any regulatory authority, and its output must not be used for "
    "clinical decision-making. Do not enter identifiable patient data."
)

data = load_model()
model = data["model"]
model_name = data.get("model_name", "RSF")
means = data["scaler_means"]
stds = data["scaler_stds"]
sel_names = data["sel_names"]
sel_vars = data["sel_vars"]
meta = data.get("meta", {})
n_patients = len(data["y_ref"])
n_events = int(data["y_ref"]["retenosis"].sum())

var_info = {}
for v in sel_vars:
    if v in means:
        var_info[v] = {"type": "cont", "label": v}
    else:
        cols = [sn for sn in sel_names if sn == v or sn.startswith(v + "_")]
        var_info[v] = {"type": "binary", "levels": [0, 1], "label": v} if len(cols) == 1 else {
            "type": "binary", "levels": [0, 1], "label": v}

LABELS = {
    "rws_percent": "RWS (% of lumen diameter)",
    "rws": "RWS (mm)",
    "rwsqfr_delta": "RWS-QFR delta",
    "QFR_stent": "QFR after stenting",
    "stenosis_percent_stent": "Residual stenosis after stenting (%)",
    "delta_qfr": "Delta QFR (stent - distal)",
    "ES_QFR": "ES-QFR",
}

st.sidebar.header("Patient parameters")
st.sidebar.caption(f"Model: {model_name} | {n_patients} patients, {n_events} events")
st.sidebar.caption(f"{len(sel_vars)} predictors retained by stability selection")

inputs = {}
cols = st.sidebar.columns(2)
for i, v in enumerate(sel_vars):
    info = var_info[v]
    label = LABELS.get(v, v)
    with cols[i % 2]:
        if info["type"] == "cont":
            mu = means.get(v, 0.0)
            sg = max(stds.get(v, 1.0), 1e-8)
            inputs[v] = st.number_input(
                label, value=float(round(mu, 3)),
                step=float(max(0.001, round(sg / 5, 3))), format="%.3f", key=v,
                help=f"cohort mean {mu:.3f}, SD {sg:.3f}",
            )
        else:
            inputs[v] = st.selectbox(label, [0, 1], key=v)


def build_input():
    x = np.zeros(len(sel_names))
    for j, sn in enumerate(sel_names):
        if sn in inputs:
            val = inputs[sn]
            if sn in means:
                val = (val - means[sn]) / stds[sn]
            x[j] = val
        else:
            for inp_name, inp_val in inputs.items():
                if sn == f"{inp_name}_{int(inp_val)}":
                    x[j] = 1.0
                    break
    return x


@st.cache_resource
def get_explainer():
    import shap

    X_ref = np.asarray(data["X_ref"])[:50]
    return shap.KernelExplainer(model.predict, X_ref)


if st.sidebar.button("Predict ISR risk", type="primary", use_container_width=True):
    x_input = build_input()
    surv = model.predict_survival_function(x_input.reshape(1, -1), return_array=True)[0]
    times = np.asarray(model.unique_times_)
    k2 = int(np.argmin(np.abs(times - 730.0)))
    prob = 1.0 - float(surv[k2])
    prob_pct = prob * 100.0

    col1, col2, col3 = st.columns(3)
    col1.metric("2-year ISR probability", f"{prob_pct:.1f}%")
    level = "Low risk" if prob_pct < 20 else ("Moderate risk" if prob_pct < 40 else "High risk")
    col2.metric("Risk category", level)
    col3.metric("Cohort 2-year event rate", "42.7%")
    st.caption(
        "The risk categories are for orientation only; they were not derived from a "
        "prospective threshold analysis."
    )

    st.subheader("Why this prediction")
    try:
        explainer = get_explainer()
        sv = explainer.shap_values(x_input.reshape(1, -1), nsamples=200, silent=True)
        X_disp = x_input.reshape(1, -1).copy()
        for j, sn in enumerate(sel_names):
            if sn in means:
                X_disp[0, j] = X_disp[0, j] * stds[sn] + means[sn]
        fig = plt.figure(figsize=(9, 4.5))
        import shap

        shap.waterfall_plot(
            shap.Explanation(
                values=np.asarray(sv).reshape(-1),
                base_values=float(np.ravel(explainer.expected_value)[0]),
                data=X_disp[0],
                feature_names=sel_names,
            ),
            max_display=10,
            show=False,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "Values are contributions to the model risk score, not to an absolute "
            "probability. Reference distribution: 50 randomly drawn cohort patients."
        )
    except Exception as e:  # pragma: no cover
        st.warning(f"SHAP explanation unavailable: {e}")

with st.expander("Model information"):
    st.write(f"**Model:** {model_name}, {meta.get('rsf_params', {})}")
    st.write(f"**Training data:** {n_patients} patients after PCI, {n_events} ISR events")
    st.write(f"**Predictors:** {sel_vars}")
    st.write(
        "**Selection:** LASSO (lambda.min, 5-fold) refitted in each of 300 nested bootstrap "
        "resamples; predictors retained at selection frequency >= 0.65."
    )
    st.write(
        "**Validation:** nested .632+ bootstrap, 300 resamples. Random survival forest "
        "out-of-bag C-index 0.659 (95% percentile interval 0.578 to 0.744); 1-year AUC 0.740."
    )
    st.write(
        "**Disclaimer:** single-centre retrospective cohort, no external validation. "
        "Not a medical device."
    )
