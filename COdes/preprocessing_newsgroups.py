"""
Text Preprocessing Pipeline for 20 NewsGroups Dataset
Implements: Text Normalization, Tokenization, Stop Word Removal, Stemming, Lemmatization
"""

import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

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


class NewsGroupsPreprocessor:
    """Preprocessing pipeline for 20 NewsGroups Dataset"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_text_from_newsgroup(self, content):
        """
        Extract the main text content from newsgroup message
        Remove headers and metadata
        """
        # Split into lines
        lines = content.split('\n')
        
        # Find where the actual content starts (after headers)
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '':
                content_start = i + 1
                break
        
        # Join the content lines
        text = '\n'.join(lines[content_start:])
        
        return text
    
    def text_normalization(self, text):
        """
        Normalize text: lowercase, remove special characters, extra spaces
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove newsgroup headers patterns
        text = re.sub(r'(from:|subject:|organization:|lines:|nntp-posting-host:).*', '', text, flags=re.IGNORECASE)
        
        # Remove special characters and digits (keep letters and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?;:\'\"-]', ' ', text)
        
        # Remove extra whitespace
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
    
    def preprocess_complete(self, text, extract_content=True):
        """
        Complete preprocessing pipeline
        """
        # Extract main content if needed
        if extract_content:
            text = self.extract_text_from_newsgroup(text)
        
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


def process_newsgroups_dataset(data_dir='20 NewsGroups', 
                               output_path='20 NewsGroups/newsgroups_preprocessed.csv',
                               sample_size=None):
    """
    Process all 20 NewsGroups text files
    """
    print("Loading 20 NewsGroups dataset...")
    
    preprocessor = NewsGroupsPreprocessor()
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if sample_size:
        txt_files = txt_files[:sample_size]
    
    print(f"Processing {len(txt_files)} newsgroup files...")
    
    results = []
    
    for txt_file in txt_files:
        file_path = os.path.join(data_dir, txt_file)
        category = txt_file.replace('.txt', '')
        
        print(f"Processing {txt_file}...")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split into individual posts (if multiple)
        # Take first 1000 characters for demonstration
        sample_content = content[:2000]
        
        # Preprocess
        processed = preprocessor.preprocess_complete(sample_content)
        
        results.append({
            'category': category,
            'original_sample': sample_content[:500],  # First 500 chars
            'normalized': processed['normalized'][:500],
            'tokens': ' '.join(processed['tokens'][:50]),  # First 50 tokens
            'filtered_tokens': ' '.join(processed['filtered_tokens'][:50]),
            'porter_stemmed': ' '.join(processed['porter_stemmed'][:50]),
            'snowball_stemmed': ' '.join(processed['snowball_stemmed'][:50]),
            'lemmatized': ' '.join(processed['lemmatized'][:50])
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    return df


def demonstrate_preprocessing():
    """
    Demonstrate preprocessing on sample newsgroup message
    """
    preprocessor = NewsGroupsPreprocessor()
    
    # Sample newsgroup message
    sample_text = """From: mathew <mathew@mantis.co.uk>
Subject: Alt.Atheism FAQ: Atheist Resources

Archive-name: atheism/resources
Alt-atheism-archive-name: resources
Last-modified: 11 December 1992
Version: 1.0

                              Atheist Resources

                      Addresses of Atheist Organizations

                                     USA

FREEDOM FROM RELIGION FOUNDATION

Darwin fish bumper stickers and assorted other atheist paraphernalia are
available from the Freedom From Religion Foundation in the US.

Write to:  FFRF, P.O. Box 750, Madison, WI 53701.
Telephone: (608) 256-8900"""
    
    print("="*80)
    print("20 NEWSGROUPS PREPROCESSING DEMONSTRATION")
    print("="*80)
    
    results = preprocessor.preprocess_complete(sample_text)
    
    print("\n1. ORIGINAL TEXT (first 200 chars):")
    print(results['original'][:200])
    
    print("\n2. NORMALIZED TEXT:")
    print(results['normalized'][:300])
    
    print("\n3. TOKENIZATION (first 30 tokens):")
    print(results['tokens'][:30])
    
    print("\n4. STOP WORD REMOVAL (first 30 tokens):")
    print(results['filtered_tokens'][:30])
    
    print("\n5. PORTER STEMMING (first 30 tokens):")
    print(results['porter_stemmed'][:30])
    
    print("\n6. SNOWBALL STEMMING (first 30 tokens):")
    print(results['snowball_stemmed'][:30])
    
    print("\n7. LEMMATIZATION (first 30 tokens):")
    print(results['lemmatized'][:30])
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Demonstrate preprocessing
    demonstrate_preprocessing()
    
    # Process NewsGroups dataset
    print("\nProcessing 20 NewsGroups dataset...")
    process_newsgroups_dataset()
    
    print("\nPreprocessing complete!")
