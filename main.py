import argparse 
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
import emoji
from textblob import TextBlob

# Download NLTK data
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])

def fix_spelling(text):
    return str(TextBlob(text).correct())

def preprocess_text(text, flags):
    """Clean and preprocess text for analysis based on flags"""
    if not isinstance(text, str) or not text.strip():
        return ""

    if flags.get('fix_spelling'):
        text = fix_spelling(text)

    text = emoji.demojize(text)

    if flags.get('remove_urls'):
        text = remove_urls(text)

    if flags.get('remove_html'):
        text = remove_html(text)

    if flags.get('remove_numbers'):
        text = remove_numbers(text)

    if flags.get('remove_hashtags'):
        text = remove_hashtags(text)

    if flags.get('remove_mentions'):
        text = remove_mentions(text)

    if flags.get('remove_punctuation'):
        text = remove_punctuation(text)

    text = text.lower()

    if flags.get('remove_stopwords'):
        text = remove_stopwords(text)

    return text.strip()

def analyze_sentiment(text):
    """Classify text sentiment as positive, negative, or neutral"""
    if not text or not isinstance(text, str):
        return "neutral"

    score = TextBlob(text).sentiment.polarity
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def extract_category(text):
    """Extract category based on keywords in text using regex"""
    if not text or not isinstance(text, str):
        return "Unknown"

    text_lower = text.lower()

    # Define patterns for categories
    patterns = {
        'Liberal': r'\b(liberal|democrat|left|progressive|joe biden|kamala harris|obama|clinton)\b',
        'Conservative': r'\b(conservative|republican|right|trump|ron desantis|ted cruz|mike pence)\b',
        'General Politics': r'\b(politics|political|election|government|congress|senate|house|president)\b',
        'World News': r'\b(world|international|foreign|ukraine|russia|china|europe)\b',
    }

    for category, pattern in patterns.items():
        if re.search(pattern, text_lower):
            return category

    return 'Other'

def main():
    parser = argparse.ArgumentParser(description="Configurable preprocessing pipeline for Reddit data")
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')

    # Cleaning flags
    parser.add_argument('--fix_spelling', action='store_true', help='Apply spelling correction')
    parser.add_argument('--remove_urls', action='store_true', help='Remove URLs')
    parser.add_argument('--remove_html', action='store_true', help='Remove HTML tags')
    parser.add_argument('--remove_numbers', action='store_true', help='Remove numbers')
    parser.add_argument('--remove_hashtags', action='store_true', help='Remove hashtags')
    parser.add_argument('--remove_mentions', action='store_true', help='Remove mentions')
    parser.add_argument('--remove_punctuation', action='store_true', help='Remove punctuation')
    parser.add_argument('--remove_stopwords', action='store_true', help='Remove stopwords')

    # Analysis flags
    parser.add_argument('--analyze_sentiment', action='store_true', help='Perform sentiment analysis')
    parser.add_argument('--extract_tags', action='store_true', help='Extract category tags using regex')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Determine if posts or comments based on columns
    if 'comment_body' in df.columns:
        # Comments dataset
        df['comment_clean'] = df['comment_body'].apply(lambda x: preprocess_text(x, vars(args)))
        clean_col = 'comment_clean'
    else:
        # Posts dataset
        df['title_clean'] = df['title'].apply(lambda x: preprocess_text(x, vars(args)))
        df['selftext_clean'] = df['selftext'].apply(lambda x: preprocess_text(x, vars(args)))
        df['full_text_clean'] = df['title_clean'] + ' ' + df['selftext_clean']
        df['full_text_clean'] = df['full_text_clean'].str.strip()
        clean_col = 'full_text_clean'

    # Apply analysis if requested
    if args.analyze_sentiment:
        df['sentiment'] = df[clean_col].apply(analyze_sentiment)

    if args.extract_tags:
        df['category'] = df[clean_col].apply(extract_category)

    # Save enhanced dataset
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()

