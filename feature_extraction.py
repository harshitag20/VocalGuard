import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Pitch (Fundamental Frequency)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    pitch_mean = np.nanmean(f0)
    pitch_std = np.nanstd(f0)

    # Jitter proxy (pitch instability)
    f0_clean = f0[~np.isnan(f0)]
    jitter_proxy = np.mean(np.abs(np.diff(f0_clean))) if len(f0_clean) > 1 else 0

    # Shimmer proxy (amplitude instability)
    rms = librosa.feature.rms(y=y)[0]
    shimmer_proxy = np.std(rms)

    # Harmonic vs noise proxy
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    hnr_proxy = np.mean(spectral_contrast)

    # Speech rate proxy
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    speech_rate = len(onsets) / librosa.get_duration(y=y, sr=sr)

    # Pause ratio
    energy = librosa.feature.rms(y=y)[0]
    pause_ratio = np.sum(energy < np.mean(energy) * 0.5) / len(energy)

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "jitter": jitter_proxy,
        "shimmer": shimmer_proxy,
        "hnr": hnr_proxy,
        "speech_rate": speech_rate,
        "pause_ratio": pause_ratio
    }
