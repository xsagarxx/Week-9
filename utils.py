import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_tweet_text(tweet):
    """
    This function takes a tweet as input and preprocesses the text for sentiment analysis.
    It removes stop words, punctuation, and converts text to lowercase.

    Parameters:
        tweet (str): A string representing a tweet.

    Returns:
        str: A preprocessed version of the tweet text.
    """
    # remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # remove user mentions
    tweet = re.sub(r'\@\w+|\#', '', tweet)

    # remove punctuation
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # tokenize tweet
    tokens = word_tokenize(tweet)

    # remove stopwords and lowercase tokens
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words("english")]

    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # join tokens back into a string
    preprocessed_tweet = " ".join(tokens)

    return preprocessed_tweet


def load_stopwords():
    """
    This function loads the stopwords used for preprocessing the tweet text.

    Returns:
        set: A set containing the stopwords.
    """
    stop_words = set(stopwords.words("english"))
    return stop_words
