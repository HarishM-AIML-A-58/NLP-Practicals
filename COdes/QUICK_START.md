# 🚀 QUICK START GUIDE - NLP Text Preprocessing

## ⚡ Get Started in 3 Steps

### Step 1: Install Dependencies (30 seconds)
```bash
pip install nltk pandas
```

### Step 2: Run Demonstration (1 minute)
```bash
python3 preprocessing_all_datasets.py --demo
```

### Step 3: Process Your Data (2 minutes)
```bash
# Process all datasets with samples
python3 preprocessing_all_datasets.py --dataset all
```

**Done! ✅** Your preprocessing is complete!

---

## 📚 What's Included?

### ✅ 5 Preprocessing Techniques
1. **Text Normalization** - Clean and standardize text
2. **Tokenization** - Split into words/sentences  
3. **Stop Word Removal** - Remove common words
4. **Stemming** - Reduce words to root form
5. **Lemmatization** - Convert to dictionary form

### ✅ 4 Dataset Processors
1. **Brown Corpus** - 57,000 sentences
2. **UD English TreeBank** - 16,000 sentences
3. **20 NewsGroups** - 20 categories
4. **Reuters** - 163,000 articles

---

## 🎯 Common Tasks

### Test Individual Techniques
```bash
python3 "Text Normalization.py"   # See normalization in action
python3 Tokenization.py            # Learn tokenization
python3 Stemming.py                # Compare stemmers
python3 Lemmatization.py           # Understand lemmas
```

### Process Specific Dataset
```bash
# Brown Corpus (100 samples)
python3 preprocessing_all_datasets.py --dataset brown --sample 100

# Reuters (100 articles)
python3 preprocessing_all_datasets.py --dataset reuters --sample 100

# UD TreeBank (50 sentences)
python3 preprocessing_all_datasets.py --dataset ud --sample 50

# 20 NewsGroups (all categories)
python3 preprocessing_all_datasets.py --dataset newsgroups
```

### Use in Your Code
```python
from preprocessing_brown_corpus import BrownCorpusPreprocessor

# Create preprocessor
preprocessor = BrownCorpusPreprocessor()

# Your text
text = "The runners were running quickly in the marathon."

# Process it
results = preprocessor.preprocess_complete(text)

# Get results
print(results['normalized'])       # Clean text
print(results['filtered_tokens'])  # Without stop words
print(results['lemmatized'])       # Dictionary forms
```

---

## 📖 Documentation

- **START HERE**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - What's implemented
- **FULL GUIDE**: [PREPROCESSING_README.md](PREPROCESSING_README.md) - Complete documentation
- **STRUCTURE**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - File organization
- **EXAMPLES**: [usage_examples.py](usage_examples.py) - Usage guide

---

## 🎓 Example: Complete Preprocessing

```python
Input:
"The RUNNERS were running QUICKLY! Visit http://example.com for more info."

Step 1 - Normalized:
"the runners were running quickly visit for more info"

Step 2 - Tokenized:
['the', 'runners', 'were', 'running', 'quickly', 'visit', 'for', 'more', 'info']

Step 3 - Stop Words Removed:
['runners', 'running', 'quickly', 'visit', 'info']

Step 4 - Stemmed:
['runner', 'run', 'quickli', 'visit', 'info']

Step 5 - Lemmatized:
['runner', 'running', 'quickly', 'visit', 'info']
```

---

## ❓ Troubleshooting

### NLTK Data Not Found?
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Import Error?
```bash
pip install nltk pandas
```

### Need Help?
Check [PREPROCESSING_README.md](PREPROCESSING_README.md) Section: "Common Issues & Solutions"

---

## 📊 Output Files

After processing, find your results in:
```
Brown Corpus/brown_preprocessed.csv
UD_English-EWT-master/ud_preprocessed.txt
20 NewsGroups/newsgroups_preprocessed.csv
Reuters/reuters_preprocessed.csv
```

---

## 🎯 Command Reference

```bash
# View help
python3 preprocessing_all_datasets.py --help

# Demo all techniques
python3 preprocessing_all_datasets.py --demo

# Process all datasets
python3 preprocessing_all_datasets.py --dataset all

# Process with custom sample size
python3 preprocessing_all_datasets.py --dataset brown --sample 500

# Individual technique demos
python3 Tokenization.py
python3 Stemming.py
python3 Lemmatization.py
```

---

## ✨ Key Features

✅ Complete preprocessing pipeline  
✅ 4 major NLP datasets supported  
✅ Multiple stemming algorithms  
✅ POS-aware lemmatization  
✅ Command-line interface  
✅ Comprehensive documentation  
✅ Production-ready code  

---

## 🚀 Ready?

```bash
# One command to see everything:
python3 preprocessing_all_datasets.py --demo
```

**That's it!** You now have a complete text preprocessing toolkit.

For more details, see [PREPROCESSING_README.md](PREPROCESSING_README.md)

---

**Exercise 1: Category 1 - Text Cleaning and Preprocessing ✅ COMPLETE**
