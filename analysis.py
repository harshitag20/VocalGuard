import os
import numpy as np
from feature_extraction import extract_features

def compute_baseline(baseline_dir):
    vectors = []

    for file in os.listdir(baseline_dir):
        path = os.path.join(baseline_dir, file)
        feats = extract_features(path)
        vectors.append(list(feats.values()))

    vectors = np.array(vectors)
    baseline_mean = np.mean(vectors, axis=0)
    baseline_std = np.std(vectors, axis=0)

    return baseline_mean, baseline_std


def compute_deviation(test_features, baseline_mean, baseline_std):
    test_vector = np.array(list(test_features.values()))
    z_scores = np.abs((test_vector - baseline_mean) / baseline_std)
    deviation_score = np.mean(z_scores)

    return deviation_score, z_scores
