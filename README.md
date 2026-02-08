#  VocalGuard

**Voice based early screening support system for Parkinson’s related speech deviations**

VocalGuard is a prototype web application that analyzes speech to detect deviations in acoustic patterns commonly associated with Parkinsonian speech. Instead of diagnosing disease, the system compares new speech samples against a baseline voice profile and flags statistically significant deviations that may warrant clinical screening.


##  Live Demo

🔗 **Web App:**  
[https://vocalguard.streamlit.app/]


##  Problem Statement

Parkinson’s disease often presents subtle speech changes years before visible motor symptoms. These early indicators are frequently overlooked due to limited access to specialists and the absence of scalable screening tools. VocalGuard aims to provide a low-cost, accessible, speech-based screening support system to identify meaningful voice deviations early.


##  Architecture Overview

The project follows a clean separation between frontend and backend logic:

| File | Role |
|-----|-----|
| `app.py` | Frontend — Streamlit web interface, input handling, visualization |
| `feature_extraction.py` | Backend — Audio signal processing and feature extraction |
| `analysis.py` | Backend — Baseline modeling and deviation scoring |
| `data/` | Audio samples (baseline, test, live uploads) |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |


##  Methodology (ML Approach)

VocalGuard uses an **unsupervised learning approach** based on baseline-driven anomaly detection.

1. **Baseline Construction**
   - Acoustic features are extracted from baseline speech samples
   - Feature-wise mean and standard deviation define a reference voice profile

2. **Feature Extraction**
   - For each new speech sample, the same acoustic features are extracted
   - Features include pitch statistics, jitter and shimmer proxies, harmonic-noise balance, speech rate, and pause ratio

3. **Deviation Scoring**
   - Each feature is compared against the baseline using normalized distance (z-score)
   - An overall deviation score is computed by aggregating feature-wise deviations

This approach focuses on **speech change over time**, not disease classification.

> **Important:** The system does not diagnose Parkinson’s. It flags speech deviations that may be clinically relevant for screening or monitoring.

##  Folder Structure
VocalGuard/
├── app.py
├── feature_extraction.py
├── analysis.py
├── data/
│ ├── baseline/ # Reference speech samples
│ ├── test/ # Test audio samples
│ └── live/ # Runtime uploads (.gitkeep placeholder)
├── requirements.txt
├── logo.png
├── README.md


Note: The `data/live` folder contains a `.gitkeep` file so it appears on GitHub. Actual recordings are generated at runtime.

##  How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/harshitag20/VocalGuard.git
cd VocalGuard

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Install FFmpeg (Required for Live Recording)

Install FFmpeg and add it to your system PATH.
Live recording requires FFmpeg when running locally.

### 4. Run the Application
python -m streamlit run app.py

## Web Deployment Notes
- On local deployment, live voice recording is supported (requires FFmpeg).

- On Streamlit Cloud, live recording is disabled due to platform limitations.

- Users can upload .wav files instead; the analysis pipeline remains identical.

## Accoustic Features Used
The following features are extracted using Librosa:

- Pitch Mean & Pitch Variability

- Jitter Proxy (pitch instability)

- Shimmer Proxy (amplitude instability)

- Harmonic–Noise Balance (HNR proxy)

- Speech Rate Proxy

- Pause Ratio

These features are widely studied in Parkinson’s speech research.

## Disclaimer
VocalGuard is a research and screening support prototype only.
It does not provide medical diagnosis or treatment recommendations.
Clinical decisions must be made by qualified healthcare professionals.

## Future Work
-Longitudinal personal speech tracking

-Integration of supervised learning with labeled datasets

-Secure user profiles and data storage

-Mobile application support