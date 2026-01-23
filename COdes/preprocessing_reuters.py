"""
Text Preprocessing Pipeline for Reuters Dataset
Implements: Text Normalization, Tokenization, Stop Word Removal, Stemming, Lemmatization
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class ReutersPreprocessor:
    """Preprocessing pipeline for Reuters Dataset"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def text_normalization(self, text):
        """
        Normalize text: lowercase, remove special characters, extra spaces
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep letters and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?;:\'\"-]', ' ', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenization(self, text):
        """
        Tokenize text into words
        """
        if not text:
            return []
        
        # Word tokenization
        tokens = word_tokenize(text)
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stop words from tokenized text
        """
        if not tokens:
            return []
        
        # Remove stopwords and single character tokens
        filtered_tokens = [word for word in tokens 
                          if word.lower() not in self.stop_words 
                          and len(word) > 1
                          and word not in string.punctuation]
        
        return filtered_tokens
    
    def stem_text(self, tokens, stemmer_type='porter'):
        """
        Apply stemming to tokens
        stemmer_type: 'porter' or 'snowball'
        """
        if not tokens:
            return []
        
        stemmer = self.porter_stemmer if stemmer_type == 'porter' else self.snowball_stemmer
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        
        return stemmed_tokens
    
    def lemmatize_text(self, tokens):
        """
        Apply lemmatization to tokens
        """
        if not tokens:
            return []
        
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return lemmatized_tokens
    
    def preprocess_complete(self, text):
        """
        Complete preprocessing pipeline
        """
        # Step 1: Text Normalization
        normalized_text = self.text_normalization(text)
        
        # Step 2: Tokenization
        tokens = self.tokenization(normalized_text)
        
        # Step 3: Remove Stop Words
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Step 4: Stemming
        porter_stemmed = self.stem_text(filtered_tokens, 'porter')
        snowball_stemmed = self.stem_text(filtered_tokens, 'snowball')
        
        # Step 5: Lemmatization
        lemmatized = self.lemmatize_text(filtered_tokens)
        
        return {
            'original': text,
            'normalized': normalized_text,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'porter_stemmed': porter_stemmed,
            'snowball_stemmed': snowball_stemmed,
            'lemmatized': lemmatized
        }


def process_reuters_dataset(csv_path='Reuters/ModApte_train.csv', 
                            output_path='Reuters/reuters_preprocessed.csv',
                            sample_size=None):
    """
    Process Reuters CSV file
    """
    print("Loading Reuters dataset...")
    
    try:
        # Try reading with different encodings
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    if sample_size:
        df = df.head(sample_size)
    
    print(f"Processing {len(df)} records...")
    
    preprocessor = ReutersPreprocessor()
    
    # Apply preprocessing to text column
    results = df['text'].apply(lambda x: preprocessor.preprocess_complete(x))
    
    # Extract results into separate columns
    df['normalized_text'] = results.apply(lambda x: x['normalized'])
    df['tokens'] = results.apply(lambda x: ' '.join(x['tokens'][:100]))  # Limit to first 100 tokens
    df['filtered_tokens'] = results.apply(lambda x: ' '.join(x['filtered_tokens'][:100]))
    df['porter_stemmed'] = results.apply(lambda x: ' '.join(x['porter_stemmed'][:100]))
    df['snowball_stemmed'] = results.apply(lambda x: ' '.join(x['snowball_stemmed'][:100]))
    df['lemmatized'] = results.apply(lambda x: ' '.join(x['lemmatized'][:100]))
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    return df


def process_all_reuters_files(data_dir='Reuters', output_dir='Reuters/preprocessed'):
    """
    Process all Reuters CSV files in the directory
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of Reuters files
    reuters_files = [
        'ModApte_train.csv',
        'ModApte_test.csv',
        'ModLewis_train.csv',
        'ModLewis_test.csv',
        'ModHayes_train.csv',
        'ModHayes_test.csv'
    ]
    
    for file_name in reuters_files:
        input_path = os.path.join(data_dir, file_name)
        output_path = os.path.join(output_dir, f'preprocessed_{file_name}')
        
        if os.path.exists(input_path):
            print(f"\nProcessing {file_name}...")
            process_reuters_dataset(input_path, output_path, sample_size=50)
        else:
            print(f"File not found: {input_path}")


def demonstrate_preprocessing():
    """
    Demonstrate preprocessing on sample Reuters text
    """
    preprocessor = ReutersPreprocessor()
    
    # Sample Reuters text
    sample_text = """Showers continued throughout the week in
the Bahia cocoa zone, alleviating the drought since early
January and improving prospects for the coming temporao,
although normal humidity levels have not been restored,
Comissaria Smith said in its weekly review."""
    
    print("="*80)
    print("REUTERS DATASET PREPROCESSING DEMONSTRATION")
    print("="*80)
    
    results = preprocessor.preprocess_complete(sample_text)
    
    print("\n1. ORIGINAL TEXT:")
    print(results['original'])
    
    print("\n2. NORMALIZED TEXT:")
    print(results['normalized'])
    
    print("\n3. TOKENIZATION:")
    print(results['tokens'])
    
    print("\n4. STOP WORD REMOVAL:")
    print(results['filtered_tokens'])
    
    print("\n5. PORTER STEMMING:")
    print(results['porter_stemmed'])
    
    print("\n6. SNOWBALL STEMMING:")
    print(results['snowball_stemmed'])
    
    print("\n7. LEMMATIZATION:")
    print(results['lemmatized'])
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Demonstrate preprocessing
    demonstrate_preprocessing()
    
    # Process Reuters dataset (sample of 100 records for demo)
    print("\nProcessing Reuters dataset...")
    process_reuters_dataset(sample_size=100)
    
    print("\nPreprocessing complete!")
