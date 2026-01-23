# NLP Practicals - Text Preprocessing

## Category 1: Text Cleaning and Preprocessing

This project implements comprehensive text preprocessing techniques for four major NLP datasets:

### Datasets
1. **Brown Corpus** - A diverse collection of American English text samples
2. **Universal Dependencies English TreeBank** - Annotated syntactic structures in CoNLL-U format
3. **20 NewsGroups** - Collection of newsgroup documents across 20 categories
4. **Reuters** - News articles for text classification tasks

### Preprocessing Techniques Implemented

#### 1. Text Normalization
- Converts text to lowercase
- Removes URLs and email addresses
- Removes special characters and digits
- Normalizes whitespace

#### 2. Tokenization
- Word tokenization - splits text into words
- Sentence tokenization - splits text into sentences
- Handles punctuation and special cases

#### 3. Stop Word Removal
- Removes common words (the, is, at, which, etc.)
- Filters out single-character tokens
- Removes punctuation marks
- Configurable stop word lists

#### 4. Stemming
Reduces words to their root form by removing suffixes:
- **Porter Stemmer** - Most widely used, moderate approach
- **Snowball Stemmer** - Improved Porter stemmer
- **Lancaster Stemmer** - Most aggressive stemming

Example: running, runs, runner → run

#### 5. Lemmatization
Reduces words to their dictionary base form using vocabulary:
- More accurate than stemming
- Context-aware with POS tagging
- Returns valid dictionary words

Example: better → good, running → run, was → be

---

## Project Structure

```
NLP-Practicals/
├── Text Normalization.py          # Text normalization implementation
├── Tokenization.py                # Tokenization techniques
├── Stop Word Removal.py           # Stop word removal
├── Stemming.py                    # Stemming algorithms
├── Lemmatization.py               # Lemmatization implementation
├── preprocessing_brown_corpus.py  # Brown Corpus preprocessing
├── preprocessing_ud_treebank.py   # UD TreeBank preprocessing
├── preprocessing_newsgroups.py    # 20 NewsGroups preprocessing
├── preprocessing_reuters.py       # Reuters preprocessing
├── preprocessing_all_datasets.py  # Unified preprocessing pipeline
└── README.md                      # This file
```

---

## Installation

### Prerequisites
```bash
# Install required packages
pip install nltk pandas

# Or use requirements.txt (create if needed)
pip install -r requirements.txt
```

### NLTK Data Downloads
The scripts automatically download required NLTK data, but you can manually download:

```python
import nltk
nltk.download('punkt')          # Tokenizer
nltk.download('stopwords')      # Stop words
nltk.download('wordnet')        # Lemmatizer dictionary
nltk.download('omw-1.4')       # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger')  # POS tagger
```

---

## Usage

### 1. Individual Technique Demonstrations

Run individual preprocessing technique demonstrations:

```bash
# Text Normalization
python "Text Normalization.py"

# Tokenization
python Tokenization.py

# Stop Word Removal
python "Stop Word Removal.py"

# Stemming
python Stemming.py

# Lemmatization
python Lemmatization.py
```

### 2. Dataset-Specific Processing

#### Brown Corpus
```bash
python preprocessing_brown_corpus.py
```

#### UD English TreeBank
```bash
python preprocessing_ud_treebank.py
```

#### 20 NewsGroups
```bash
python preprocessing_newsgroups.py
```

#### Reuters Dataset
```bash
python preprocessing_reuters.py
```

### 3. Unified Processing Pipeline

#### Run demonstration on all datasets:
```bash
python preprocessing_all_datasets.py
# or
python preprocessing_all_datasets.py --demo
```

#### Process all datasets:
```bash
python preprocessing_all_datasets.py --dataset all
```

#### Process specific dataset:
```bash
# Brown Corpus (100 samples)
python preprocessing_all_datasets.py --dataset brown --sample 100

# UD TreeBank (50 sentences)
python preprocessing_all_datasets.py --dataset ud --sample 50

# 20 NewsGroups (all categories)
python preprocessing_all_datasets.py --dataset newsgroups

# Reuters (100 articles)
python preprocessing_all_datasets.py --dataset reuters --sample 100
```

---

## Output Files

After processing, the following output files are generated:

```
Brown Corpus/brown_preprocessed.csv          # Preprocessed Brown Corpus
UD_English-EWT-master/ud_preprocessed.txt    # Preprocessed UD TreeBank
20 NewsGroups/newsgroups_preprocessed.csv    # Preprocessed NewsGroups
Reuters/reuters_preprocessed.csv             # Preprocessed Reuters
```

Each output file contains:
- Original text
- Normalized text
- Tokens
- Filtered tokens (after stop word removal)
- Porter stemmed tokens
- Snowball stemmed tokens
- Lemmatized tokens

---

## Code Examples

### Example 1: Using Individual Techniques

```python
from preprocessing_brown_corpus import BrownCorpusPreprocessor

# Initialize preprocessor
preprocessor = BrownCorpusPreprocessor()

# Sample text
text = "The runners were running quickly in the marathon."

# 1. Normalize
normalized = preprocessor.text_normalization(text)
print(normalized)  # "the runners were running quickly in the marathon"

# 2. Tokenize
tokens = preprocessor.tokenization(normalized)
print(tokens)  # ['the', 'runners', 'were', 'running', 'quickly', 'in', 'the', 'marathon']

# 3. Remove stop words
filtered = preprocessor.remove_stopwords(tokens)
print(filtered)  # ['runners', 'running', 'quickly', 'marathon']

# 4. Stem
stemmed = preprocessor.stem_text(filtered)
print(stemmed)  # ['runner', 'run', 'quickli', 'marathon']

# 5. Lemmatize
lemmatized = preprocessor.lemmatize_text(filtered)
print(lemmatized)  # ['runner', 'running', 'quickly', 'marathon']
```

### Example 2: Complete Pipeline

```python
from preprocessing_brown_corpus import BrownCorpusPreprocessor

preprocessor = BrownCorpusPreprocessor()
text = "The runners were running quickly in the marathon."

# Run complete pipeline
results = preprocessor.preprocess_complete(text)

print("Original:", results['original'])
print("Normalized:", results['normalized'])
print("Tokens:", results['tokens'])
print("Filtered:", results['filtered_tokens'])
print("Stemmed:", results['porter_stemmed'])
print("Lemmatized:", results['lemmatized'])
```

### Example 3: Processing Custom Text

```python
from preprocessing_reuters import ReutersPreprocessor

preprocessor = ReutersPreprocessor()

# Your custom text
custom_text = """
Breaking News: Scientists have discovered a new species of butterfly 
in the Amazon rainforest. The discovery was made during an expedition 
conducted by researchers from various universities.
"""

# Preprocess
results = preprocessor.preprocess_complete(custom_text)

# Access results
print("Normalized:", results['normalized'])
print("Key terms:", results['filtered_tokens'])
print("Stemmed:", results['porter_stemmed'])
```

---

## Comparison: Stemming vs Lemmatization

| Word | Porter Stemmer | Lemmatizer |
|------|---------------|------------|
| studies | studi | study |
| studying | study | studying |
| better | better | good |
| running | run | running |
| was | wa | be |
| caring | care | caring |

**When to use:**
- **Stemming**: Fast, good for search/IR, doesn't need POS tags
- **Lemmatization**: Accurate, preserves meaning, needs more computation

---

## Performance Considerations

### Processing Time (approximate for 1000 documents)
- Text Normalization: ~1 second
- Tokenization: ~2 seconds
- Stop Word Removal: ~1 second
- Stemming: ~3 seconds
- Lemmatization: ~15 seconds (slower but more accurate)

### Memory Usage
- Small datasets (< 10K docs): < 100MB
- Medium datasets (10K-100K docs): 100MB - 1GB
- Large datasets (> 100K docs): > 1GB

---

## Technical Details

### Text Normalization Rules
1. Convert to lowercase: `"Hello World" → "hello world"`
2. Remove URLs: `"Visit http://example.com" → "Visit"`
3. Remove emails: `"Contact test@example.com" → "Contact"`
4. Keep: letters, spaces, basic punctuation (.,!?;:'\"-)
5. Remove: digits, special characters (@#$%^&*)
6. Normalize whitespace: `"too  many   spaces" → "too many spaces"`

### Stop Words
English stop words (179 words including):
- Articles: a, an, the
- Prepositions: in, on, at, for, to
- Pronouns: i, you, he, she, it, we, they
- Common verbs: is, am, are, was, were, be, been
- Others: and, or, but, if, then, etc.

---

## Dataset Information

### 1. Brown Corpus
- **Format**: CSV with columns: filename, para_id, sent_id, raw_text, tokenized_text, tokenized_pos, label
- **Size**: ~57,000 rows
- **Categories**: Religion, fiction, news, learned, etc.
- **Use Case**: General text analysis, genre classification

### 2. Universal Dependencies English TreeBank
- **Format**: CoNLL-U (tab-separated)
- **Size**: ~247,000 lines (~16K sentences)
- **Features**: Tokens, lemmas, POS tags, dependencies
- **Use Case**: Syntactic analysis, dependency parsing

### 3. 20 NewsGroups
- **Format**: Text files (one per category)
- **Size**: ~120,000 lines per file
- **Categories**: 20 (comp.graphics, rec.sport.baseball, sci.med, etc.)
- **Use Case**: Text classification, topic modeling

### 4. Reuters
- **Format**: CSV with columns: text, text_type, topics, places, etc.
- **Size**: ~163,000 rows (train set)
- **Categories**: Multiple topics (earnings, acquisitions, etc.)
- **Use Case**: News classification, information retrieval

---

## Common Issues & Solutions

### Issue 1: NLTK Data Not Found
```
Error: Resource punkt not found
```
**Solution:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Issue 2: Encoding Errors
```
Error: UnicodeDecodeError
```
**Solution:** The scripts handle this automatically with fallback encodings.

### Issue 3: Memory Issues with Large Datasets
**Solution:** Use the `sample_size` parameter:
```bash
python preprocessing_all_datasets.py --dataset reuters --sample 1000
```

---

## Future Enhancements

- [ ] Add more languages support
- [ ] Implement custom stop word lists
- [ ] Add n-gram generation
- [ ] Include POS tagging in preprocessing
- [ ] Add TF-IDF calculation
- [ ] Implement parallel processing for large datasets
- [ ] Add visualization of preprocessing results
- [ ] Create web interface for easy testing

---

## References

1. **NLTK Documentation**: https://www.nltk.org/
2. **Brown Corpus**: Francis, W. Nelson, and Henry Kučera (1979)
3. **Universal Dependencies**: https://universaldependencies.org/
4. **20 NewsGroups**: http://qwone.com/~jason/20Newsgroups/
5. **Reuters Dataset**: Lewis et al. (1997) Reuters-21578

---

## License

This project is for educational purposes as part of NLP Practicals coursework.

---

## Contact

For questions or issues, please refer to the course materials or contact the instructor.

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install nltk pandas

# 2. Run demonstration
python preprocessing_all_datasets.py --demo

# 3. Process all datasets
python preprocessing_all_datasets.py --dataset all

# 4. Test individual techniques
python Tokenization.py
python Stemming.py
python Lemmatization.py
```

Happy preprocessing! 🎯
