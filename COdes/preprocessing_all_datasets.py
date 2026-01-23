"""
Unified Text Preprocessing Pipeline for All Datasets
Implements: Text Normalization, Tokenization, Stop Word Removal, Stemming, Lemmatization

Datasets:
1. Brown Corpus
2. Universal Dependencies - English TreeBank
3. 20 NewsGroups
4. Reuters

Usage:
    python preprocessing_all_datasets.py --dataset all
    python preprocessing_all_datasets.py --dataset brown --sample 100
    python preprocessing_all_datasets.py --dataset ud --sample 50
    python preprocessing_all_datasets.py --dataset newsgroups
    python preprocessing_all_datasets.py --dataset reuters --sample 100
"""

import argparse
import sys
from preprocessing_brown_corpus import process_brown_corpus, BrownCorpusPreprocessor
from preprocessing_ud_treebank import process_ud_treebank, UDTreeBankPreprocessor
from preprocessing_newsgroups import process_newsgroups_dataset, NewsGroupsPreprocessor
from preprocessing_reuters import process_reuters_dataset, ReutersPreprocessor


def demonstrate_all_preprocessing():
    """
    Demonstrate preprocessing techniques on all datasets
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE TEXT PREPROCESSING DEMONSTRATION")
    print("Category 1: Text Cleaning and Preprocessing")
    print("="*80)
    
    # 1. Brown Corpus
    print("\n" + "="*80)
    print("DATASET 1: BROWN CORPUS")
    print("="*80)
    preprocessor_brown = BrownCorpusPreprocessor()
    sample_brown = "Furthermore, as an encouragement to revisionist thinking, it manifestly is fair to admit that any fraternity has a constitutional right."
    result_brown = preprocessor_brown.preprocess_complete(sample_brown)
    
    print("\nOriginal:", result_brown['original'])
    print("Normalized:", result_brown['normalized'])
    print("Tokens:", result_brown['tokens'][:15])
    print("Filtered:", result_brown['filtered_tokens'][:15])
    print("Stemmed:", result_brown['porter_stemmed'][:15])
    print("Lemmatized:", result_brown['lemmatized'][:15])
    
    # 2. UD TreeBank
    print("\n" + "="*80)
    print("DATASET 2: UNIVERSAL DEPENDENCIES - ENGLISH TREEBANK")
    print("="*80)
    preprocessor_ud = UDTreeBankPreprocessor()
    sample_ud = {
        'text': "American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque.",
        'tokens': [], 'lemmas': [], 'pos_tags': []
    }
    result_ud = preprocessor_ud.preprocess_sentence(sample_ud)
    
    print("\nOriginal:", result_ud['original'])
    print("Normalized:", result_ud['normalized'])
    print("Tokens:", result_ud['tokens'][:15])
    print("Filtered:", result_ud['filtered_tokens'][:15])
    print("Stemmed:", result_ud['porter_stemmed'][:15])
    print("Lemmatized:", result_ud['lemmatized'][:15])
    
    # 3. 20 NewsGroups
    print("\n" + "="*80)
    print("DATASET 3: 20 NEWSGROUPS")
    print("="*80)
    preprocessor_news = NewsGroupsPreprocessor()
    sample_news = """From: mathew@mantis.co.uk
Subject: Atheist Resources

Darwin fish bumper stickers and assorted atheist paraphernalia are available from the Freedom From Religion Foundation."""
    result_news = preprocessor_news.preprocess_complete(sample_news)
    
    print("\nOriginal (excerpt):", result_news['original'][:80])
    print("Normalized:", result_news['normalized'][:80])
    print("Tokens:", result_news['tokens'][:15])
    print("Filtered:", result_news['filtered_tokens'][:15])
    print("Stemmed:", result_news['porter_stemmed'][:15])
    print("Lemmatized:", result_news['lemmatized'][:15])
    
    # 4. Reuters
    print("\n" + "="*80)
    print("DATASET 4: REUTERS")
    print("="*80)
    preprocessor_reuters = ReutersPreprocessor()
    sample_reuters = "Showers continued throughout the week in the Bahia cocoa zone, alleviating the drought since early January and improving prospects."
    result_reuters = preprocessor_reuters.preprocess_complete(sample_reuters)
    
    print("\nOriginal:", result_reuters['original'])
    print("Normalized:", result_reuters['normalized'])
    print("Tokens:", result_reuters['tokens'][:15])
    print("Filtered:", result_reuters['filtered_tokens'][:15])
    print("Stemmed:", result_reuters['porter_stemmed'][:15])
    print("Lemmatized:", result_reuters['lemmatized'][:15])
    
    print("\n" + "="*80)
    print("PREPROCESSING TECHNIQUES DEMONSTRATED:")
    print("1. Text Normalization - Lowercase, remove special chars, clean whitespace")
    print("2. Tokenization - Split text into words")
    print("3. Stop Word Removal - Remove common words (the, is, at, etc.)")
    print("4. Stemming - Reduce words to root form (running -> run)")
    print("5. Lemmatization - Reduce words to dictionary form (better -> good)")
    print("="*80 + "\n")


def process_all_datasets(sample_sizes=None):
    """
    Process all datasets
    """
    if sample_sizes is None:
        sample_sizes = {
            'brown': 100,
            'ud': 50,
            'newsgroups': None,
            'reuters': 100
        }
    
    print("\n" + "="*80)
    print("PROCESSING ALL DATASETS")
    print("="*80)
    
    # Process Brown Corpus
    print("\n1. Processing Brown Corpus...")
    try:
        process_brown_corpus(sample_size=sample_sizes['brown'])
        print("✓ Brown Corpus processing complete")
    except Exception as e:
        print(f"✗ Error processing Brown Corpus: {e}")
    
    # Process UD TreeBank
    print("\n2. Processing UD English TreeBank...")
    try:
        process_ud_treebank(sample_size=sample_sizes['ud'])
        print("✓ UD TreeBank processing complete")
    except Exception as e:
        print(f"✗ Error processing UD TreeBank: {e}")
    
    # Process NewsGroups
    print("\n3. Processing 20 NewsGroups...")
    try:
        process_newsgroups_dataset(sample_size=sample_sizes['newsgroups'])
        print("✓ 20 NewsGroups processing complete")
    except Exception as e:
        print(f"✗ Error processing 20 NewsGroups: {e}")
    
    # Process Reuters
    print("\n4. Processing Reuters...")
    try:
        process_reuters_dataset(sample_size=sample_sizes['reuters'])
        print("✓ Reuters processing complete")
    except Exception as e:
        print(f"✗ Error processing Reuters: {e}")
    
    print("\n" + "="*80)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("="*80 + "\n")


def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(description='Text Preprocessing Pipeline for NLP Datasets')
    parser.add_argument('--dataset', type=str, default='all',
                      choices=['all', 'brown', 'ud', 'newsgroups', 'reuters', 'demo'],
                      help='Dataset to process (default: all)')
    parser.add_argument('--sample', type=int, default=None,
                      help='Sample size (number of records to process)')
    parser.add_argument('--demo', action='store_true',
                      help='Run demonstration on sample texts')
    
    args = parser.parse_args()
    
    # Run demonstration if requested
    if args.demo or args.dataset == 'demo':
        demonstrate_all_preprocessing()
        return
    
    # Process specific dataset or all
    if args.dataset == 'all':
        sample_sizes = {
            'brown': args.sample or 100,
            'ud': args.sample or 50,
            'newsgroups': None,
            'reuters': args.sample or 100
        }
        process_all_datasets(sample_sizes)
    
    elif args.dataset == 'brown':
        print("\nProcessing Brown Corpus...")
        process_brown_corpus(sample_size=args.sample or 100)
    
    elif args.dataset == 'ud':
        print("\nProcessing UD English TreeBank...")
        process_ud_treebank(sample_size=args.sample or 50)
    
    elif args.dataset == 'newsgroups':
        print("\nProcessing 20 NewsGroups...")
        process_newsgroups_dataset(sample_size=args.sample)
    
    elif args.dataset == 'reuters':
        print("\nProcessing Reuters...")
        process_reuters_dataset(sample_size=args.sample or 100)


if __name__ == "__main__":
    # If no arguments provided, run demonstration
    if len(sys.argv) == 1:
        demonstrate_all_preprocessing()
        print("\nTo process datasets, use:")
        print("  python preprocessing_all_datasets.py --dataset all")
        print("  python preprocessing_all_datasets.py --dataset brown --sample 100")
        print("  python preprocessing_all_datasets.py --demo")
    else:
        main()
