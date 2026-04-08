import nltk
from nltk import ngrams
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


class ProtectedAttributesStopWordsCheck(object):
    def __init__(self, denied_phrases_file):
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("punkt")
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stemmed_lemmatized_stop_list = set()
        stop_list = list()
        with open(denied_phrases_file, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                stop_list.append(line)
        for phrase in stop_list:
            tokens = self.tokenizer.tokenize(phrase)
            tokens = [self.stemmer.stem(self.lemmatizer.lemmatize(token)) for token in tokens]
            self.stemmed_lemmatized_stop_list.add(" ".join(tokens))

    def check_phrase_is_stoplist_compliant(self, query):
        """
        Checks entire query against phrases in stoplist file.
        This is used as a check prior to classifier model to block
        clearly explicit words or phrases.
        """
        unigrams = self.tokenizer.tokenize(query.lower())
        unigrams = [self.stemmer.stem(self.lemmatizer.lemmatize(token)) for token in unigrams]

        # Exact match check for stoplist phrases in sentence
        unigrams_str = " " + " ".join(unigrams) + " "
        for phrase in self.stemmed_lemmatized_stop_list:
            phrase_with_space = " " + phrase + " "
            if phrase_with_space in unigrams_str:
                return False

        # Unigram and bigram checks
        for unigram in unigrams:
            if unigram in self.stemmed_lemmatized_stop_list:
                return False
        bigrams = list(ngrams(unigrams, 2))
        for bigram in bigrams:
            if " ".join(bigram) in self.stemmed_lemmatized_stop_list:
                return False
        return True

    def get_sentences(self, text):
        return nltk.sent_tokenize(text)

    @classmethod
    def get_instance(cls, denied_phrases_file):
        if not hasattr(cls, "instance"):
            cls.instance = ProtectedAttributesStopWordsCheck(
                denied_phrases_file=denied_phrases_file
            )
        return cls.instance
