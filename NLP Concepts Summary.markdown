# NLP Concepts Summary

## 1. Lemmatization
**Definition:** Reduces words to their base or dictionary form (lemma).  
**Purpose:** Groups different forms of a word into one base form for better analysis.  
**Example:**  
- running, ran, runs → run  
- better → good (contextual)  
**Note:** Uses part-of-speech tagging for accuracy.

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required data once
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer

# Lemmatize verb 'running' to base form 'run'
print(lemmatizer.lemmatize('running', pos='v'))  # Output: run

# Lemmatize adjective 'better' to base form 'good'
print(lemmatizer.lemmatize('better', pos='a'))  # Output: good
```

## 2. Stop Words
**Definition:** Common words with little semantic meaning (e.g., the, is, at).  
**Purpose:** Removed to reduce noise and improve efficiency in text processing.  
**Example:**  
- Original: "This is a simple example."  
- After removal: "This simple example."

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required data once
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))  # Load English stop words
sentence = "This is a simple example."  # Input sentence

words = word_tokenize(sentence)  # Tokenize sentence into words

# Filter out stop words
filtered = [word for word in words if word.lower() not in stop_words]

print(filtered)  # Output: ['This', 'simple', 'example', '.']
```

## 3. Bag of Words (BOW)
**Definition:** Converts text into a vector based on word counts.  
**Purpose:** Provides a simple numeric representation ignoring word order.  
**Example:**  
- Sentences:  
  - "I love cats"  
  - "I love dogs"  
- Vocabulary: [I, love, cats, dogs]  
- Vector for sentence 1: [1, 1, 1, 0]  
- Vector for sentence 2: [1, 1, 0, 1]

```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = ["I love cats", "I love dogs"]  # Input sentences

vectorizer = CountVectorizer()  # Initialize BOW vectorizer

bow = vectorizer.fit_transform(sentences)  # Learn vocabulary and vectorize sentences

print(vectorizer.get_feature_names_out())  
# Output: ['cats' 'dogs' 'love' 'i'] — vocabulary sorted alphabetically

print(bow.toarray())  
# Output: 
# [[1 0 1 1]  -> "I love cats"
#  [0 1 1 1]] -> "I love dogs"
```

## 4. TF-IDF (Term Frequency - Inverse Document Frequency)
**Definition:** Weighs words based on frequency in a document and rarity across documents.  
**Purpose:** Highlights important words while down-weighting common ones.  
**Formula:**  
- TF = (Frequency of term in document) / (Total terms in document)  
- IDF = log (Total documents / Documents containing term)  
- TF-IDF = TF × IDF  
**Example:**  
- Word "cats" appears frequently in one document but rarely in others → high TF-IDF.  
- Word "I" appears in all documents → low TF-IDF.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = ["I love cats", "I love dogs"]  # Input sentences

tfidf_vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer

tfidf = tfidf_vectorizer.fit_transform(sentences)  # Learn vocabulary and compute TF-IDF scores

print(tfidf_vectorizer.get_feature_names_out())  
# Output: ['cats' 'dogs' 'love' 'i']

print(tfidf.toarray())  
# Output: TF-IDF weighted vectors, e.g.,
# [[0.70710678 0.         0.70710678 0.        ]
#  [0.         0.70710678 0.70710678 0.        ]]
```