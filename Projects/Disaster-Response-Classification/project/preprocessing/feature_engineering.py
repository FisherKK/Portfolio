import re
import nltk

for package in ["punkt", "wordnet", "stopwords"]:
    nltk.download(package)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


def tokenize(text):
    """Cleans the string by: cast string to lowercase, leaves only letters, disjoints string into list of tokens,
    removes stopwords, lemmatizes and stemmes tokens, removes additional space. Returns lists of cleaned tokens.

    Parameters:
    -----------
    text: string
        String to be tokenized.
    """
    raw_text = re.sub(r"[^a-zA-Z]", " ", text.lower())

    tokens = word_tokenize(raw_text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = tok.strip()
        clean_tok = stemmer.stem(clean_tok)
        clean_tok = lemmatizer.lemmatize(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens
