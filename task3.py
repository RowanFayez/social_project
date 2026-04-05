"""
Task 3: Sentiment Analysis & Exploratory Data Analysis (EDA)
Performs sentiment analysis on preprocessed Reddit political data.

This script uses:
  - VADER (NLTK SentimentIntensityAnalyzer) — optimized for social media text
  - TextBlob — polarity & subjectivity scores
  - Pandas for statistical summaries

Usage:
    python task3.py --input cleaned.csv --output sentiment_results.csv
    python task3.py --input cleaned.csv --output sentiment_results.csv --text_column full_text_clean
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ─── NLTK data ───────────────────────────────────────────────────────────────

def _ensure_vader():
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)

# ─── Sentiment helpers ────────────────────────────────────────────────────────

def vader_scores(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """Return VADER compound, positive, neutral, negative scores."""
    if not isinstance(text, str) or not text.strip():
        return {'vader_compound': 0.0, 'vader_pos': 0.0,
                'vader_neu': 1.0, 'vader_neg': 0.0}
    scores = sia.polarity_scores(text)
    return {
        'vader_compound': round(scores['compound'], 4),
        'vader_pos':      round(scores['pos'],      4),
        'vader_neu':      round(scores['neu'],      4),
        'vader_neg':      round(scores['neg'],      4),
    }


def vader_label(compound: float) -> str:
    """Convert VADER compound score to a sentiment label."""
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'


def textblob_scores(text: str) -> dict:
    """Return TextBlob polarity and subjectivity scores."""
    if not isinstance(text, str) or not text.strip():
        return {'tb_polarity': 0.0, 'tb_subjectivity': 0.0}
    blob = TextBlob(text)
    return {
        'tb_polarity':     round(blob.sentiment.polarity,     4),
        'tb_subjectivity': round(blob.sentiment.subjectivity, 4),
    }


def textblob_label(polarity: float) -> str:
    """Convert TextBlob polarity to a sentiment label."""
    if polarity > 0.05:
        return 'positive'
    if polarity < -0.05:
        return 'negative'
    return 'neutral'


# ─── Core analysis ────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """Sentiment analysis pipeline for Reddit political post data."""

    def __init__(self):
        _ensure_vader()
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Add VADER and TextBlob sentiment columns to *df*.

        Columns added:
            vader_compound, vader_pos, vader_neu, vader_neg, vader_label
            tb_polarity, tb_subjectivity, tb_label
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        print(f"\nRunning VADER sentiment analysis on '{text_column}'...")
        vader_data = df[text_column].apply(
            lambda t: vader_scores(t, self.sia)
        )
        vader_df = pd.DataFrame(vader_data.tolist(), index=df.index)
        vader_df['vader_label'] = vader_df['vader_compound'].apply(vader_label)

        print("Running TextBlob sentiment analysis...")
        tb_data = df[text_column].apply(textblob_scores)
        tb_df = pd.DataFrame(tb_data.tolist(), index=df.index)
        tb_df['tb_label'] = tb_df['tb_polarity'].apply(textblob_label)

        result = pd.concat([df, vader_df, tb_df], axis=1)
        return result

    # ── Reporting ──────────────────────────────────────────────────────────────

    @staticmethod
    def print_summary(df: pd.DataFrame):
        """Print a human-readable analysis report to stdout."""
        sep = "=" * 65

        print(f"\n{sep}")
        print("TASK 3 — SENTIMENT ANALYSIS REPORT")
        print(sep)
        print(f"Dataset size : {len(df):,} posts")

        # ── VADER distribution ────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("VADER SENTIMENT DISTRIBUTION")
        print(f"{'─'*65}")
        vader_counts = df['vader_label'].value_counts()
        total = len(df)
        for label in ['positive', 'neutral', 'negative']:
            count = vader_counts.get(label, 0)
            pct   = count / total * 100
            bar   = '█' * int(pct / 2)
            print(f"  {label:<10} {count:>4} ({pct:5.1f}%)  {bar}")

        print(f"\n  Avg compound score : {df['vader_compound'].mean():.4f}")
        print(f"  Std compound score : {df['vader_compound'].std():.4f}")

        # ── TextBlob distribution ─────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print("TEXTBLOB SENTIMENT DISTRIBUTION")
        print(f"{'─'*65}")
        tb_counts = df['tb_label'].value_counts()
        for label in ['positive', 'neutral', 'negative']:
            count = tb_counts.get(label, 0)
            pct   = count / total * 100
            bar   = '█' * int(pct / 2)
            print(f"  {label:<10} {count:>4} ({pct:5.1f}%)  {bar}")

        print(f"\n  Avg polarity     : {df['tb_polarity'].mean():.4f}")
        print(f"  Avg subjectivity : {df['tb_subjectivity'].mean():.4f}")

        # ── By category ───────────────────────────────────────────────────────
        if 'category' in df.columns:
            print(f"\n{'─'*65}")
            print("VADER SENTIMENT BY CATEGORY")
            print(f"{'─'*65}")
            grp = df.groupby('category')['vader_compound'].agg(
                Count='count',
                Mean='mean',
                Std='std',
            ).round(4)
            grp['Sentiment'] = grp['Mean'].apply(vader_label)
            print(grp.to_string())

        # ── Top posts ─────────────────────────────────────────────────────────
        if 'title' in df.columns:
            print(f"\n{'─'*65}")
            print("TOP 5 MOST POSITIVE POSTS (VADER compound)")
            print(f"{'─'*65}")
            top_pos = df.nlargest(5, 'vader_compound')[
                ['title', 'vader_compound', 'category']
            ] if 'category' in df.columns else df.nlargest(5, 'vader_compound')[
                ['title', 'vader_compound']
            ]
            for _, row in top_pos.iterrows():
                cat = f" [{row['category']}]" if 'category' in row else ""
                print(f"  [{row['vader_compound']:+.4f}]{cat} {str(row['title'])[:90]}")

            print(f"\n{'─'*65}")
            print("TOP 5 MOST NEGATIVE POSTS (VADER compound)")
            print(f"{'─'*65}")
            top_neg = df.nsmallest(5, 'vader_compound')[
                ['title', 'vader_compound', 'category']
            ] if 'category' in df.columns else df.nsmallest(5, 'vader_compound')[
                ['title', 'vader_compound']
            ]
            for _, row in top_neg.iterrows():
                cat = f" [{row['category']}]" if 'category' in row else ""
                print(f"  [{row['vader_compound']:+.4f}]{cat} {str(row['title'])[:90]}")

        # ── Agreement between methods ─────────────────────────────────────────
        if 'vader_label' in df.columns and 'tb_label' in df.columns:
            agree = (df['vader_label'] == df['tb_label']).sum()
            pct   = agree / total * 100
            print(f"\n{'─'*65}")
            print("MODEL AGREEMENT")
            print(f"{'─'*65}")
            print(f"  VADER & TextBlob agree on {agree}/{total} posts ({pct:.1f}%)")

        print(f"\n{sep}\n")


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 3: Sentiment analysis on preprocessed Reddit political data"
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to input CSV (e.g. cleaned.csv from Task 2)'
    )
    parser.add_argument(
        '--output', required=True,
        help='Path for output CSV with sentiment columns'
    )
    parser.add_argument(
        '--text_column', default='full_text_clean',
        help='Column to run sentiment analysis on (default: full_text_clean)'
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nLoading data from '{input_path}'...")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"Loaded {len(df):,} rows with columns: {df.columns.tolist()}")

    # Fall back to 'title' if the requested column is absent; raise clearly if
    # neither the requested column nor 'title' exists.
    text_col = args.text_column
    if text_col not in df.columns:
        if not df.columns.tolist():
            raise ValueError("Input DataFrame has no columns.")
        fallback = 'title' if 'title' in df.columns else None
        if fallback is None:
            raise ValueError(
                f"Column '{text_col}' not found and no 'title' column to fall back to. "
                f"Available columns: {df.columns.tolist()}"
            )
        print(
            f"Warning: column '{text_col}' not found; "
            f"falling back to '{fallback}'"
        )
        text_col = fallback

    # ── Run analysis ──────────────────────────────────────────────────────────
    analyzer = SentimentAnalyzer()
    result_df = analyzer.analyze(df, text_col)

    # ── Print report ──────────────────────────────────────────────────────────
    SentimentAnalyzer.print_summary(result_df)

    # ── Save output ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to '{output_path}'")
    new_cols = [c for c in result_df.columns if c not in df.columns]
    print(f"New columns added: {new_cols}\n")


if __name__ == "__main__":
    main()
