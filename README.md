# Social Analytics Project

---

## Task 3: Sentiment Analysis & EDA

### Overview

`task3.py` performs sentiment analysis on the preprocessed Reddit political data produced by
Task 2. It adds eight new columns to the dataset and prints a human-readable report to the
terminal.

Two independent methods are used and compared:

| Method | Library | Scores |
|--------|---------|--------|
| **VADER** | NLTK `SentimentIntensityAnalyzer` | compound, pos, neu, neg |
| **TextBlob** | TextBlob | polarity, subjectivity |

VADER is specifically tuned for social-media text (handles punctuation, ALL CAPS, etc.) and is
the primary classifier; TextBlob provides a second opinion and a subjectivity dimension.

### Usage

```bash
# Run on the cleaned output from Task 2
python task3.py --input cleaned.csv --output sentiment_results.csv

# Choose a different text column (default: full_text_clean)
python task3.py --input cleaned.csv --output sentiment_results.csv --text_column title_clean
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ✓ | — | Path to input CSV (output of Task 2) |
| `--output` | ✓ | — | Path for output CSV |
| `--text_column` | | `full_text_clean` | Column to analyse |

### Output Columns Added

| Column | Type | Description |
|--------|------|-------------|
| `vader_compound` | float [-1, 1] | Overall VADER sentiment score |
| `vader_pos` | float [0, 1] | Proportion of positive tone |
| `vader_neu` | float [0, 1] | Proportion of neutral tone |
| `vader_neg` | float [0, 1] | Proportion of negative tone |
| `vader_label` | str | `positive` / `neutral` / `negative` (compound ≥ 0.05 / ≤ -0.05) |
| `tb_polarity` | float [-1, 1] | TextBlob polarity |
| `tb_subjectivity` | float [0, 1] | TextBlob subjectivity (0 = objective, 1 = subjective) |
| `tb_label` | str | `positive` / `neutral` / `negative` (polarity > 0.05 / < -0.05) |

### Terminal Report

The script prints:

1. VADER and TextBlob sentiment distributions with bar charts
2. Average compound score and subjectivity
3. VADER sentiment broken down by political **category** (if the column is present)
4. Top 5 most-positive and most-negative posts by compound score
5. Agreement rate between the two methods

### No Extra Dependencies

`task3.py` relies only on packages already listed in `pyproject.toml`:
`pandas`, `nltk` (VADER lexicon auto-downloaded on first run), and `textblob`.

---

## Task 2: Robust Text Preprocessing & Data Refinement

## Overview

This is a configurable text preprocessing pipeline for social media data (Reddit political discussions). The pipeline provides modular cleaning steps that can be toggled via command-line arguments.

## Features

- **Modular Design**: Each preprocessing step is a separate function
- **CLI-Driven**: All options are controlled via command-line flags
- **Default Off**: All cleaning flags are `False` by default
- **Category Extraction**: Regex-based categorization for data partitioning
- **Advanced NLP**: Lemmatization and spelling correction support

## Installation

### Required Dependencies

```bash
pip install pandas emoji nltk textblob
```

### NLTK Data

The script will automatically download required NLTK data (stopwords, wordnet) when needed.

## Usage

### Basic Syntax

```bash
python main.py --input <input_file.csv> --output <output_file.csv> [OPTIONS]
```

### Available Options

#### Input/Output

- `--input INPUT` - Input CSV file path (required)
- `--output OUTPUT` - Output CSV file path (required)

#### Basic Cleaning

- `--remove_urls` - Remove HTTP/HTTPS URLs
- `--remove_html` - Remove HTML tags
- `--remove_numbers` - Remove numeric digits
- `--remove_hashtags` - Remove hashtags (#tag)
- `--remove_mentions` - Remove @ mentions
- `--remove_reddit_formatting` - Remove Reddit markdown formatting
- `--remove_punctuation` - Remove punctuation marks
- `--remove_emojis` - Convert emojis to text descriptions
- `--lowercase` - Convert text to lowercase

#### Advanced Options

- `--remove_stopwords` - Remove common English stopwords
- `--fix_spelling` - Apply spelling correction using TextBlob
- `--lemmatize` - Apply lemmatization using TextBlob (Word → Base Form)

#### Analysis Features

- `--extract_tags` - Extract category/subject tags using regex patterns

#### Convenience

- `--all` - Apply all preprocessing steps

## Examples

### Example 1: Minimal Preprocessing

```bash
python main.py --input reddit_political_posts.csv \
               --output cleaned_posts.csv \
               --lowercase \
               --remove_urls
```

### Example 2: Standard Text Preprocessing

```bash
python main.py --input reddit_political_comments.csv \
               --output cleaned_comments.csv \
               --lowercase \
               --remove_urls \
               --remove_punctuation \
               --remove_stopwords \
               --extract_tags
```

### Example 3: Advanced NLP Pipeline

```bash
python main.py --input reddit_political_posts.csv \
               --output posts_advanced.csv \
               --lowercase \
               --remove_urls \
               --remove_punctuation \
               --lemmatize \
               --extract_tags
```

### Example 4: Full Preprocessing (All Options)

```bash
python main.py --input raw_data.csv \
               --output fully_cleaned.csv \
               --all
```

### Example 5: Spelling Correction + Tagging Only

```bash
python main.py --input reddit_posts.csv \
               --output corrected_posts.csv \
               --fix_spelling \
               --extract_tags
```

## Input Data Format

The script expects CSV files with the following columns:

### For Posts Dataset:

- `title` - Post title (required)
- `selftext` - Post body text (optional)
- `subreddit` - Subreddit name (used for categorization)
- Other columns are preserved

### For Comments Dataset:

- `comment_body` - Comment text (required)
- `subreddit` - Subreddit name (used for categorization)
- `post_id` - Parent post ID (optional)
- Other columns are preserved

## Output Format

The script adds cleaned versions of text columns:

- `title_clean` - Cleaned version of `title`
- `selftext_clean` - Cleaned version of `selftext`
- `comment_body_clean` - Cleaned version of `comment_body`
- `full_text_clean` - Combined cleaned title + selftext (for posts)
- `category` - Extracted category tag (if `--extract_tags` is used)

All original columns are preserved in the output.

## Category Extraction

When `--extract_tags` is enabled, the script extracts categories based on:

1. **Subreddit names** (primary method)
   - Uses the existing `subreddit` column as the category

2. **Political topic keywords** (fallback method)
   - `election`: voting, ballot, campaign
   - `policy`: legislation, bill, law, reform
   - `economy`: economic, inflation, unemployment, jobs
   - `healthcare`: medicare, medicaid, insurance
   - `immigration`: immigrant, border, visa
   - `climate`: environment, warming, carbon
   - `foreign_policy`: international, diplomacy, war
   - `other`: unclassified

## Justification for Preprocessing Choices

### What This Pipeline Includes

- **URL Removal**: URLs add noise and don't contribute to sentiment/topic analysis
- **Lowercase**: Normalizes text for consistent analysis
- **Lemmatization**: Reduces words to base form (running → run) without losing meaning
- **Category Tags**: Essential for granular analysis across different political topics

### What This Pipeline Avoids (By Default)

- **Stopword Removal**: Optional because stopwords can be important for sentiment:
  - "not good" vs "good" - negation is critical
  - "but", "however" - sentiment shifters
- **Aggressive Cleaning**: Preserves context needed for political discourse analysis

### When to Use What

- **Sentiment Analysis**: Use `--lowercase --remove_urls --lemmatize`
- **Topic Modeling**: Add `--remove_stopwords --remove_numbers`
- **Named Entity Analysis**: Avoid `--lowercase` to preserve proper nouns
- **Quick Cleaning**: Use `--all` for comprehensive preprocessing

## Performance Notes

- **Spelling Correction** (`--fix_spelling`): Slow for large datasets (limits to 100 words/text)
- **Lemmatization** (`--lemmatize`): Moderate speed, requires wordnet download
- **Basic Cleaning**: Fast (regex-based operations)

## Output Statistics

The script prints:

- Total texts processed
- Successfully cleaned texts
- Empty/invalid texts
- Applied cleaning steps
- Category distribution (if tags extracted)

## Error Handling

The script gracefully handles:

- Missing columns (skips them)
- Empty/null text values
- NLTK data download failures
- File encoding issues (uses UTF-8-sig)

## Troubleshooting

### Issue: "wordnet not found"

**Solution**: The script will auto-download. Ensure internet connection.

### Issue: Slow processing with --fix_spelling

**Solution**: Spelling correction is computationally expensive. Consider processing a smaller subset or removing this flag.

### Issue: Output has fewer characters than input

**Solution**: This is expected - preprocessing removes noise. Check summary statistics to verify cleaning steps.

## Project Structure

```
social_project/
├── SocialProjectTask1.ipynb             # Task 1: Data collection via Reddit/NewsData APIs
├── main.py                              # Task 2: Preprocessing pipeline script
├── task3.py                             # Task 3: Sentiment analysis & EDA
├── README.md                            # This file
├── reddit_political_posts.csv           # Input: Posts dataset (raw)
├── cleaned.csv                          # Intermediate: Cleaned posts from Task 2
└── sentiment_results.csv                # Output: Posts with sentiment scores from Task 3
```

## Author

Social Analytics Project - Task 2

## License

Educational purposes - Social Media Analysis Course
