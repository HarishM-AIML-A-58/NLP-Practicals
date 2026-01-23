# Text Preprocessing on Brown Corpus
# Simple and direct implementation (No classes, no complexity)

import nltk
import string

from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --------------------------------------------------
# 1. Load Brown Corpus
# --------------------------------------------------
text = " ".join(brown.words())

# --------------------------------------------------
# 2. Text Normalization (Lowercasing)
# --------------------------------------------------
text = text.lower()

# --------------------------------------------------
# 3. Text Cleaning (Remove punctuation & numbers)
# --------------------------------------------------
clean_text = ""

for char in text:
    if char not in string.punctuation and not char.isdigit():
        clean_text += char

# --------------------------------------------------
# 4. Tokenization
# --------------------------------------------------
tokens = word_tokenize(clean_text)

# --------------------------------------------------
# 5. Stopword Removal
# --------------------------------------------------
stop_words = set(stopwords.words('english'))

filtered_tokens = []
for word in tokens:
    if word not in stop_words:
        filtered_tokens.append(word)

# --------------------------------------------------
# 6. Stemming
# --------------------------------------------------
stemmer = PorterStemmer()
stemmed_words = []

for word in filtered_tokens:
    stemmed_words.append(stemmer.stem(word))

# --------------------------------------------------
# 7. Lemmatization
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()
lemmatized_words = []

for word in filtered_tokens:
    lemmatized_words.append(lemmatizer.lemmatize(word))

# --------------------------------------------------
# Display Sample Output
# --------------------------------------------------
print("Original Tokens (first 20):")
print(tokens[:20])

print("\nAfter Stopword Removal (first 20):")
print(filtered_tokens[:20])

print("\nAfter Stemming (first 20):")
print(stemmed_words[:20])

print("\nAfter Lemmatization (first 20):")
print(lemmatized_words[:20])
