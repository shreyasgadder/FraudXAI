import numpy as np

def compute_sparsity(importance_values, threshold=0.01):
    importance_values = np.asarray(importance_values)

    non_zero = int((importance_values > threshold).sum())
    total = int(len(importance_values))

    return {
        "non_zero": non_zero,
        "total": total,
        "sparsity_ratio": float(non_zero / total) if total > 0 else 0.0
    }
