import spacy
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import nltk

# Download the stopwords corpus, which contains lists of commonly used words
nltk.download('stopwords')
# Download the Punkt tokenizer models (pre-trained models for tokenizing text into sentences or words)
nltk.download('punkt')
# Downloads the model required for part-of-speech tagging in NLTK
nltk.download('averaged_perceptron_tagger_eng')

# Load the English language model (en_core_web_sm) from SpaCy for natural language processing
nlp = spacy.load('en_core_web_sm')

# Extract the default list of stop words provided by the SpaCy language model
stop_words_spacy = nlp.Defaults.stop_words
# Load the list of English stop words from the NLTK library
stop_words_nltk = set(stopwords.words('english'))

def nltk_keywords(data):
    """
    Extract and return a list of keywords using NLTK.
    """
    # Tokenize the input text
    tokens = word_tokenize(data)
    # Tag each token with its part of speech
    tagged_tokens = pos_tag(tokens)

    # Extract tokens labeled as proper nouns (NNP) or common nouns (NN) as potential keywords
    keywords = [str(token[0]) for token in tagged_tokens if token[1] in ['NNP', 'NN']]
    # Filter out stop words from the list of potential keywords using NLTK's stop word list
    keywords = [word for word in keywords if word not in stop_words_nltk]
    # Remove duplicates, sort the keywords alphabetically and convert them to a list
    keywords = sorted(list(set(keyword for keyword in keywords)))
    return keywords


def spacy_keywords(data):
    """
    Extract and return a list of keywords using SpaCy.
    """
    # Pass the input text through the SpaCy model to generate token objects
    tokens = nlp(data)
    # Create a list of tuples containing a token and its part-of-speech tag
    tagged_tokens = [(tok, tok.tag_) for tok in tokens]

    # Extract tokens labeled as proper nouns (NNP) or common nouns (NN) as potential keywords
    keywords = [str(token[0]) for token in tagged_tokens if token[1] in ['NNP', 'NN']]
    # Filter out stop words from the list of potential keywords using SpaCy's stop word list
    keywords = [word for word in keywords if word not in stop_words_spacy]
    # Remove duplicates, sort the keywords alphabetically and convert them to a list
    keywords = sorted(list(set(keyword for keyword in keywords)))
    return keywords