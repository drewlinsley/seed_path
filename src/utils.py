import numpy as np
import pandas as pd


def quantize_DNA(dna):
    # Convert DNA values to numeric, replacing '<0.01' with 0
    dna_numeric = np.array([0 if x == '<0.01' else float(x) for x in dna])

    # Compute the median of non-zero values
    median = np.median(dna_numeric[dna_numeric > 0])

    # Create quantized labels:
    # 0: DNA = 0 (originally <0.01)
    # 1: 0 < DNA ≤ median
    # 2: DNA > median
    dna_quantized = np.zeros_like(dna_numeric, dtype=int)
    dna_quantized[(dna_numeric > 0) & (dna_numeric <= median)] = 1
    dna_quantized[dna_numeric > median] = 2

    # Print distribution of classes
    for i in range(3):
        print(f"Class {i}: {np.sum(dna_quantized == i)} samples")

    return dna_quantized