"""Illustrate aggregate validation on independent synthetic comparison pairs.

The design supplies exact category probabilities and fixed conditional-mean
fits. The example estimates a learning-bias allowance only; evaluation draws
and a sampling bound are additionally needed for an association certificate.
"""
from pathlib import Path
import json
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'results' / 'aggregate_bias'))
from aggregate_bias_bound import aggregate_upper_bound


def main():
    rng = random.Random(1309202631)
    masses = [0.5, 0.5]  # Known from this sampling design, not the observed counts.
    success = [0.2, 0.8]
    fits = [0.35, 0.65]  # Fixed exact conditional midrank means for this example.
    pairs_per_stream = 2048
    counts = [[0, 0], [0, 0]]
    sums = [[0.0, 0.0], [0.0, 0.0]]
    # Assign streams before generating any values. Every iteration draws a new
    # focal/reference pair. Reference categories keep their marginal law.
    for stream in [0, 1]:
        for _ in range(pairs_per_stream):
            focal_category = int(rng.random() >= masses[0])
            reference_category = int(rng.random() >= masses[0])
            focal = [int(rng.random() < success[focal_category]) for _ in [0, 1]]
            reference = [int(rng.random() < success[reference_category]) for _ in [0, 1]]
            comparison = float(focal[stream] > reference[stream]) + 0.5 * (focal[stream] == reference[stream])
            counts[stream][focal_category] += 1
            sums[stream][focal_category] += comparison - fits[focal_category]
    means = [[sums[h][c] / counts[h][c] if counts[h][c] else 0.0 for c in [0, 1]] for h in [0, 1]]
    absent = [max((u - f) * (v - f) for u in [0.0, 1.0] for v in [0.0, 1.0]) for f in fits]
    result = aggregate_upper_bound(masses, means[0], means[1], counts[0], counts[1], 0.05, absent_upper=absent)
    print(json.dumps(dict(
        scope='Synthetic demonstration of a learning-bias bound, not an association or forecasting decision.',
        category_masses=masses, stream_counts=counts,
        validation_pairs=2 * pairs_per_stream, raw_observation_calls=4 * pairs_per_stream,
        true_bias_for_this_synthetic_design=0.0, bound=result,
    ), indent=2))


if __name__ == '__main__':
    main()
