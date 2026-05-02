"""
Leave-one-out evaluation for the animal sound search index.

This script evaluates retrieval quality without letting a query match itself.
It uses the same scaled vectors as search.py.
"""

import os
import sys
from collections import defaultdict

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from database import DB_PATH, load_all_vectors
from search import normalize_l2


def evaluate_leave_one_out(top_k=5, db_path=DB_PATH):
    ids, filenames, species, matrix = load_all_vectors(db_path, scaled=True)
    if len(ids) == 0:
        raise RuntimeError("Database is empty. Run src/indexing.py first.")

    vectors = normalize_l2(matrix)
    scores = vectors @ vectors.T

    hit_at_1 = 0
    precision_sum = 0.0
    per_species = defaultdict(lambda: {'n': 0, 'hit1': 0, 'p_at_k': 0.0})

    for i, true_species in enumerate(species):
        order = np.argsort(scores[i])[::-1]
        order = [idx for idx in order if idx != i][:top_k]

        top_species = [species[idx] for idx in order]
        hit = bool(top_species and top_species[0] == true_species)
        precision = sum(sp == true_species for sp in top_species) / top_k

        hit_at_1 += int(hit)
        precision_sum += precision

        row = per_species[true_species]
        row['n'] += 1
        row['hit1'] += int(hit)
        row['p_at_k'] += precision

    total = len(ids)
    summary = {
        'total': total,
        'n_species': len(set(species)),
        'hit_at_1': hit_at_1 / total,
        f'precision_at_{top_k}': precision_sum / total,
        'per_species': {},
    }

    for sp, row in sorted(per_species.items()):
        n = row['n']
        summary['per_species'][sp] = {
            'count': n,
            'hit_at_1': row['hit1'] / n,
            f'precision_at_{top_k}': row['p_at_k'] / n,
        }

    return summary


def main():
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    summary = evaluate_leave_one_out(top_k=top_k)

    print("Leave-one-out evaluation")
    print(f"Files: {summary['total']}")
    print(f"Species: {summary['n_species']}")
    print(f"Hit@1: {summary['hit_at_1']:.4f}")
    print(f"Precision@{top_k}: {summary[f'precision_at_{top_k}']:.4f}")
    print()
    print(f"{'Species':<14} {'Files':>5} {'Hit@1':>8} {f'P@{top_k}':>8}")
    print("-" * 40)
    for sp, row in summary['per_species'].items():
        print(
            f"{sp:<14} {row['count']:>5} "
            f"{row['hit_at_1']:>8.4f} "
            f"{row[f'precision_at_{top_k}']:>8.4f}"
        )


if __name__ == '__main__':
    main()
