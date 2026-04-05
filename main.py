"""
Task 2: Robust Text Preprocessing & Data Refinement
A configurable preprocessing pipeline for Reddit political data

This script provides modular text preprocessing with command-line toggleable options.
All cleaning flags are False by default.
"""

import argparse
import re
import string
import warnings
from pathlib import Path

import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob, Word

warnings.filterwarnings("ignore")

# NLTK data will be downloaded on-demand when used
STOPWORDS = None
WORDNET_DOWNLOADED = False


class TextPreprocessor:
    """Modular text preprocessing pipeline with configurable cleaning steps"""
    
    def __init__(self, config):
        """Initialize preprocessor with configuration from CLI arguments"""
        self.config = config
        self.stats = {
            'total_texts': 0,
            'cleaned_texts': 0,
            'empty_texts': 0,
            'categories_extracted': 0
        }
    
    def remove_urls(self, text):
        """Remove HTTP/HTTPS URLs and www links"""
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    def remove_html(self, text):
        """Remove HTML tags"""
        return re.sub(r'<.*?>', '', text)
    
    def remove_numbers(self, text):
        """Remove all numeric digits"""
        return re.sub(r'\d+', '', text)
    
    def remove_hashtags(self, text):
        """Remove hashtags"""
        return re.sub(r'#\w+', '', text)
    
    def remove_mentions(self, text):
        """Remove @ mentions"""
        return re.sub(r'@\w+', '', text)
    
    def remove_reddit_formatting(self, text):
        """Remove Reddit-specific markdown and formatting"""
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove asterisks for bold/italic
        text = re.sub(r'\*+', '', text)
        # Remove underscores for emphasis
        text = re.sub(r'\_+', ' ', text)
        # Remove quote markers
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        return text
    
    def remove_punctuation(self, text):
        """Remove all punctuation marks"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_stopwords(self, text):
        """Remove common English stopwords"""
        global STOPWORDS
        if STOPWORDS is None:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading stopwords corpus...")
                nltk.download('stopwords', quiet=True)
            STOPWORDS = set(stopwords.words('english'))
        
        return " ".join([word for word in text.split() if word.lower() not in STOPWORDS])
    
    def convert_emojis(self, text):
        """Convert emojis to text descriptions"""
        return emoji.demojize(text)
    
    def fix_spelling(self, text):
        """Correct spelling errors using TextBlob"""
        try:
            # Process word by word to avoid very long processing times
            words = text.split()
            if len(words) > 100:  # Limit for performance
                words = words[:100]
            
            corrected_words = []
            for word in words:
                # Only correct words with letters and reasonable length
                if word.isalpha() and 3 <= len(word) <= 15:
                    try:
                        corrected = str(Word(word).correct())
                        corrected_words.append(corrected)
                    except:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            
            return " ".join(corrected_words)
        except Exception as e:
            print(f"Warning: Spelling correction failed: {e}")
            return text
    
    def lemmatize_text(self, text):
        """Apply lemmatization using TextBlob"""
        global WORDNET_DOWNLOADED
        try:
            # Ensure wordnet is downloaded (only once)
            if not WORDNET_DOWNLOADED:
                try:
                    nltk.data.find('corpora/wordnet')
                    WORDNET_DOWNLOADED = True
                except LookupError:
                    print("Downloading wordnet corpus for lemmatization...")
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                    WORDNET_DOWNLOADED = True
            
            words = text.split()
            lemmatized = [Word(word).lemmatize() for word in words if word]
            return " ".join(lemmatized)
        except Exception as e:
            print(f"Warning: Lemmatization failed: {e}")
            return text
    
    def lowercase_text(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def extract_category_tags(self, df, text_column='title'):
        """
        Extract category tags using regex patterns for political Reddit data.
        Categories based on subreddit names and political topics.
        """
        if 'subreddit' in df.columns:
            # Use subreddit as primary category
            df['category'] = df['subreddit'].astype(str)
            self.stats['categories_extracted'] = len(df)
            return df
        
        # Fallback: Extract political topics from text
        political_keywords = {
            'election': ['election', 'voting', 'ballot', 'campaign'],
            'policy': ['policy', 'legislation', 'bill', 'law', 'reform'],
            'economy': ['economy', 'economic', 'inflation', 'unemployment', 'jobs'],
            'healthcare': ['healthcare', 'medicare', 'medicaid', 'insurance'],
            'immigration': ['immigration', 'immigrant', 'border', 'visa'],
            'climate': ['climate', 'environment', 'warming', 'carbon'],
            'foreign_policy': ['foreign', 'international', 'diplomacy', 'war'],
            'other': []
        }
        
        def classify_text(text):
            if not isinstance(text, str):
                return 'other'
            text_lower = text.lower()
            for category, keywords in political_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return category
            return 'other'
        
        df['category'] = df[text_column].apply(classify_text)
        self.stats['categories_extracted'] = len(df[df['category'] != 'other'])
        return df
    
    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            self.stats['empty_texts'] += 1
            return ""
        
        self.stats['total_texts'] += 1
        
        if self.config.remove_emojis:
            text = self.convert_emojis(text)
        
        if self.config.remove_urls:
            text = self.remove_urls(text)
        
        if self.config.remove_html:
            text = self.remove_html(text)
        
        if self.config.remove_reddit_formatting:
            text = self.remove_reddit_formatting(text)
        
        if self.config.remove_numbers:
            text = self.remove_numbers(text)
        
        if self.config.remove_hashtags:
            text = self.remove_hashtags(text)
        
        if self.config.remove_mentions:
            text = self.remove_mentions(text)
        
        if self.config.lowercase:
            text = self.lowercase_text(text)
        
        if self.config.remove_punctuation:
            text = self.remove_punctuation(text)
        
        if self.config.fix_spelling:
            text = self.fix_spelling(text)
        
        if self.config.lemmatize:
            text = self.lemmatize_text(text)
        
        if self.config.remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Clean up extra whitespace
        text = " ".join(text.split())
        
        if text.strip():
            self.stats['cleaned_texts'] += 1
        
        return text.strip()
    
    def process_dataframe(self, df):
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE STARTING")
        print("="*60)
        
        text_columns = []
        
        if 'title' in df.columns:
            text_columns.append('title')
        if 'selftext' in df.columns:
            text_columns.append('selftext')
        
        if 'comment_body' in df.columns:
            text_columns.append('comment_body')
        
        if not text_columns:
            print("Warning: No recognized text columns found in dataset")
            return df
        
        print(f"\nText columns to process: {', '.join(text_columns)}")
        print(f"Total rows: {len(df)}")
        
        for col in text_columns:
            print(f"\nProcessing column: '{col}'...")
            clean_col_name = f"{col}_clean"
            df[clean_col_name] = df[col].apply(self.preprocess_text)
            
            # Show statistics
            original_avg_len = df[col].astype(str).str.len().mean()
            cleaned_avg_len = df[clean_col_name].str.len().mean()
            print(f"  Average length: {original_avg_len:.1f} -> {cleaned_avg_len:.1f} chars")
        
        # Create combined cleaned text for posts
        if 'title_clean' in df.columns and 'selftext_clean' in df.columns:
            df['full_text_clean'] = df['title_clean'] + " " + df['selftext_clean']
            df['full_text_clean'] = df['full_text_clean'].str.strip()
        
        # Extract category tags if requested
        if self.config.extract_tags:
            print("\nExtracting category tags...")
            df = self.extract_category_tags(df)
            print(f"Categories assigned: {self.stats['categories_extracted']} texts")
            if 'category' in df.columns:
                print("\nCategory distribution:")
                print(df['category'].value_counts())
        
        return df
    
    def print_summary(self):
        """Print preprocessing statistics"""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total texts processed: {self.stats['total_texts']}")
        print(f"Successfully cleaned: {self.stats['cleaned_texts']}")
        print(f"Empty/invalid texts: {self.stats['empty_texts']}")
        
        print("\nCleaning steps applied:")
        for key, value in vars(self.config).items():
            if isinstance(value, bool) and value and key not in ['extract_tags']:
                print(f"  ✓ {key.replace('_', ' ').title()}")
        
        if self.config.extract_tags:
            print(f"\nCategory tags extracted: {self.stats['categories_extracted']}")
        
        print("="*60)


def parse_arguments():
    """Parse command-line arguments for configurable preprocessing"""
    parser = argparse.ArgumentParser(
        description='Task 2: Robust Text Preprocessing Pipeline for Social Media Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply all preprocessing steps
  python main.py --input reddit_political_posts.csv --output cleaned_posts.csv --all
  
  # Apply only spelling correction and lemmatization
  python main.py --input reddit_political_comments.csv --output clean_comments.csv --fix_spelling --lemmatize
  
  # Extract categories with basic cleaning
  python main.py --input reddit_political_posts.csv --output clean_posts.csv --extract_tags --lowercase --remove_urls
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    
    # Preprocessing flags (all False by default)
    parser.add_argument('--remove_urls', action='store_true',
                        help='Remove HTTP/HTTPS URLs')
    parser.add_argument('--remove_html', action='store_true',
                        help='Remove HTML tags')
    parser.add_argument('--remove_numbers', action='store_true',
                        help='Remove numeric digits')
    parser.add_argument('--remove_hashtags', action='store_true',
                        help='Remove hashtags (#tag)')
    parser.add_argument('--remove_mentions', action='store_true',
                        help='Remove @ mentions')
    parser.add_argument('--remove_reddit_formatting', action='store_true',
                        help='Remove Reddit markdown formatting')
    parser.add_argument('--remove_punctuation', action='store_true',
                        help='Remove punctuation marks')
    parser.add_argument('--remove_stopwords', action='store_true',
                        help='Remove common English stopwords')
    parser.add_argument('--remove_emojis', action='store_true',
                        help='Convert emojis to text descriptions')
    parser.add_argument('--lowercase', action='store_true',
                        help='Convert text to lowercase')
    
    # Advanced preprocessing (as per task requirements)
    parser.add_argument('--fix_spelling', action='store_true',
                        help='Apply spelling correction using TextBlob')
    parser.add_argument('--lemmatize', action='store_true',
                        help='Apply lemmatization using TextBlob')
    
    # Category extraction
    parser.add_argument('--extract_tags', action='store_true',
                        help='Extract category/subject tags using regex')
    
    # Convenience flag
    parser.add_argument('--all', action='store_true',
                        help='Apply all preprocessing steps')
    
    args = parser.parse_args()
    
    # If --all flag is set, enable all preprocessing options
    if args.all:
        args.remove_urls = True
        args.remove_html = True
        args.remove_numbers = True
        args.remove_hashtags = True
        args.remove_mentions = True
        args.remove_reddit_formatting = True
        args.remove_punctuation = True
        args.remove_stopwords = True
        args.remove_emojis = True
        args.lowercase = True
        args.fix_spelling = True
        args.lemmatize = True
        args.extract_tags = True
    
    return args


def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    print(f"\nLoading data from: {args.input}")
    
    # Load data
    try:
        df = pd.read_csv(args.input, encoding='utf-8-sig')
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(args)
    
    # Process dataframe
    df_cleaned = preprocessor.process_dataframe(df)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving cleaned data to: {args.output}")
    df_cleaned.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"✓ Successfully saved {len(df_cleaned)} rows")
    
    # Print summary
    preprocessor.print_summary()
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
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

