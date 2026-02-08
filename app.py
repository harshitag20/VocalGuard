
from analysis import compute_baseline,compute_deviation
from audiorecorder import audiorecorder
import soundfile as sf
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from feature_extraction import extract_features

st.set_page_config(page_title="VocalGuard", layout="centered")
st.title("VocalGuard")
st.subheader("Live Voice Recording(Live recording available in local deployment only)")
baseline_dir = "data/baseline"
baseline_mean, baseline_std = compute_baseline(baseline_dir)

audio = audiorecorder(
    "Start Recording",
    "Stop Recording",
    key="live_audio_recorder"
)

if len(audio) > 0:
    st.success("Recording captured")

    # Convert AudioSegment to numpy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize to [-1, 1]
    samples /= np.max(np.abs(samples))

    live_path = "data/live/live_recording.wav"
    sf.write(live_path, samples, audio.frame_rate)

    st.audio(live_path)


    if st.button("Analyze Live Recording", key="analyze_live"):
        live_features = extract_features(live_path)
        live_vector = np.array(list(live_features.values()))

        deviation_score, z_scores = compute_deviation(live_features, baseline_mean, baseline_std)

        st.subheader("Live Recording Result")
        st.metric("Deviation Score", round(deviation_score, 2))

        if deviation_score < 1:
            st.success("Clinical Indicator: Normal speech pattern")
        elif deviation_score < 2:
            st.warning("Clinical Indicator: Mild speech deviation")
        else:
            st.error("Clinical Indicator: Significant speech deviation")

        st.caption(
            "This output reflects speech deviation only and does not provide a medical diagnosis."
        )


st.caption("Clinical-style voice screening prototype")
test_dir = "data/test"
test_file = st.selectbox("Select speech sample", os.listdir(test_dir))

if st.button("Run Speech Assessment"):
    test_path = os.path.join(test_dir, test_file)
    test_feats = extract_features(test_path)
    deviation_score, z_scores = compute_deviation(test_feats, baseline_mean, baseline_std)

    st.subheader("Overall Result")
    st.metric("Deviation Score", round(deviation_score, 2))

    if deviation_score < 1:
        st.success("Clinical Indicator: Normal speech pattern")
    elif deviation_score < 2:
        st.warning("Clinical Indicator: Mild speech deviation")
    else:
        st.error("Clinical Indicator: Significant speech deviation")

    st.subheader("Clinical Feature Summary")
    for k, v in test_feats.items():
        st.write(f"{k.replace('_',' ').title()}: {round(v, 3)}")

    fig, ax = plt.subplots()
    ax.bar(test_feats.keys(), z_scores)
    ax.set_ylabel("Deviation from Baseline")
    ax.set_title("Feature-wise Deviation Analysis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.caption(
    "This tool supports early screening and research use only. It does not provide medical diagnosis."
)
