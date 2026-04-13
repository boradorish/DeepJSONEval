import os
import argparse
import pandas as pd

SCORE_COLS = ['format_score', 'detailed_score', 'strict_score']

def get_args():
    parser = argparse.ArgumentParser('Summarize DeepJSONEval evaluation results')
    parser.add_argument('--eval-path', default='eval_results', type=str,
                        help='folder containing evaluation result xlsx files')
    parser.add_argument('--by-category', action='store_true',
                        help='also show per-category breakdown')
    return parser.parse_args()


def load_results(eval_path):
    records = {}
    for fname in sorted(os.listdir(eval_path)):
        if not fname.endswith('.xlsx'):
            continue
        model = fname.replace('.xlsx', '')
        df = pd.read_excel(os.path.join(eval_path, fname))
        records[model] = df
    return records


def difficulty(df):
    medium = df[df['true_depth'].isin([3, 4])]
    hard   = df[df['true_depth'].isin([5, 6, 7])]
    return medium, hard


def mean_row(df, label):
    row = df[SCORE_COLS].mean()
    row.name = label
    row['n'] = len(df)
    return row


def print_table(title, rows):
    tbl = pd.DataFrame(rows)[['n'] + SCORE_COLS]
    tbl['n'] = tbl['n'].astype(int)
    tbl[SCORE_COLS] = (tbl[SCORE_COLS] * 100).round(2)
    print(f'\n{title}')
    print(tbl.to_string())


def main():
    args = get_args()
    records = load_results(args.eval_path)

    if not records:
        print(f'No xlsx files found in {args.eval_path}')
        return

    # ── Overall summary across all models ─────────────────────────────────────
    overall_rows = []
    for model, df in records.items():
        overall_rows.append(mean_row(df, model))
    print_table('=== Overall (scores in %) ===', overall_rows)

    # ── Per-model: Medium / Hard breakdown ────────────────────────────────────
    diff_rows = []
    for model, df in records.items():
        med, hard = difficulty(df)
        diff_rows.append(mean_row(med,  f'{model} | Medium (depth 3-4)'))
        diff_rows.append(mean_row(hard, f'{model} | Hard   (depth 5-7)'))
    print_table('=== By Difficulty (scores in %) ===', diff_rows)

    # ── Per-category (optional) ───────────────────────────────────────────────
    if args.by_category:
        for model, df in records.items():
            cat_rows = []
            for cat, grp in df.groupby('category'):
                cat_rows.append(mean_row(grp, cat))
            print_table(f'=== {model} — By Category (scores in %) ===', cat_rows)


if __name__ == '__main__':
    main()
