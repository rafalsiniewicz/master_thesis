import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
from base_class import *

class Document(Base):
    def __init__(self, name="DOCUMENT", _filename="", _id=-1, _content="", _topic=""):
        super().__init__(name)
        self.filename: str = _filename
        self.id: int = _id
        self.content: str = _content
        self.topic: str = _topic
        self.tokens: dict = dict()
        self.features: List[float] = []                 # list of all features for document
        self.tf: dict[str: float] = dict()
        self.tf_idf: dict[str: float] = dict()          # if tf_idf = 0, then the word doesnt appear in document

    def __str__(self):
        return "\n### Document ###\nfilename: {}\nid: {}\ntopic: {}\n################\n".format(self.__filename, self.__id,
                                                                                        self.__topic)

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, filename: str):
        if not (isinstance(filename, str) or filename is None):
            raise TypeError("documents filename must be a string")
        self.__filename = filename

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id: int):
        if not (isinstance(id, int) or id is None):
            raise TypeError("documents filename must be a string")
        self.__id = id

    @property
    def topic(self):
        return self.__topic

    @topic.setter
    def topic(self, topic: str):
        if not (isinstance(topic, str) or topic is None):
            raise TypeError("documents filename must be a string")
        self.__topic = topic

    @property
    def tokens(self):
        return self.__tokens

    @tokens.setter
    def tokens(self, tokens: dict):
        if not ((isinstance(tokens, dict) and all(isinstance(element, str) for element in tokens)) or tokens is None):
            raise TypeError("tokens must be a dict: tokens[term]=number_of_occurencess")
        self.__tokens = tokens

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features: List[float]):
        if not ((isinstance(features, list) and all(
                isinstance(element, float) for element in features)) or features is None):
            raise TypeError("features must be a list of floats")
        self.__features = features

    @property
    def tf(self):
        return self.__tf

    @tf.setter
    def tf(self, tf: dict):
        if not (isinstance(tf, dict) or tf is None):
            raise TypeError("TF of document must be a dict")
        self.__tf = tf

    @property
    def tf_idf(self):
        return self.__tf_idf

    @tf_idf.setter
    def tf_idf(self, tf_idf: dict):
        if not (isinstance(tf_idf, dict) or tf_idf is None):
            raise TypeError("TF-IDF of document must be a dict")
        self.__tf_idf = tf_idf

    def read_content(self, filename: str = None):
        """
        Read file content and put it in self.content.
        :param filename:        file name to read
        :return:
        """
        if not isinstance(filename, str):
            raise TypeError("Error: Filename must be a string")
        #TODO read file differently depending on file type, so .csv, .txt, no extension will be read differently
        # print(f.read())
        if self.filename != filename:
            self.filename = filename

        self.log_debug("Reading content from {}".format(self.filename))
        f = open(filename, "r", encoding="utf8")
        self.content = f.read()

    def tokenize(self) -> List[str]:
        """
        Convert document content to lower case and split it into tokens.
        :return: self.tokens (dict()):           dict of tokens for document with number of occurences per word
        """
        self.log_debug("Tokenizing {}".format(self.filename))
        if self.content and isinstance(self.content, str):
            # self.tokens = []
            # for t in re.split(' |, |\n', self.content):
            #     if t != '':
            #         self.tokens.append(t)
            self.tokens = dict(sorted(Counter(word_tokenize(self.content)).items()))
        else:
            raise ValueError("Document content is not correct")

        return self.tokens

    def remove_non_alphanumeric(self, to_lower: bool = False) -> str:
        """
        Remove all non alphanumeric characters, e.g. numbers, special characters, commas, dots, etc.
        Steps:
        1. Convert to lower case (optional)
        2. Remove email addresses
        3. Remove numbers
        4. Remove punctuation (removes this set of symbols '''!()-[]{};:'"\, <>./?@#$%^&*_~''')
        5. Remove whitespaces
        :param to_lower (bool):         if True then convert text to lower case
        :return: self.content           document content after removing all non-alphanumeric characters
        """
        if self.content and isinstance(self.content, str):

            # 1. Convert to lower case (optional)
            if to_lower:
                self.content = self.content.lower()

            # 2. Remove email addresses
            self.content = re.sub(r'[\w\.-]+@[\w\.-]+', '', self.content)

            # 3. Remove numbers
            self.content = re.sub(r'\d+', '', self.content)

            # 4. Remove punctuation
            self.content = re.sub(r'[^\w\s]', '', self.content)

            # 5. Remove whitespaces
            self.content = self.content.strip()

        else:
            raise ValueError("Document content is not correct")

        return self.content

    def remove_stop_words(self) -> List[str]:
        """
        Remove all stop words (defined for english in nltk) from self.tokens
        :return: self.tokens:               list of document's tokens
        """
        self.log_debug("Removing stop words from {}".format(self.filename))
        if self.tokens and isinstance(self.tokens, dict) and all(isinstance(element, str) for element in self.tokens):
            stop_words = set(stopwords.words('english'))
            # input(stop_words)
            for w in list(self.tokens):
                if w in stop_words:
                    self.tokens.pop(w)
        else:
            raise ValueError("Document tokens are not correct")

        return self.tokens

    def lemmatize(self) -> List[str]:
        """
        Lemmatize all tokens in self.tokens.
        :return: self.tokens:               list of document's tokens
        """

        self.log_debug("Lemmatizing tokens in {}".format(self.filename))
        if self.tokens and isinstance(self.tokens, dict) and all(isinstance(element, str) for element in self.tokens):
            lemmatizer = WordNetLemmatizer()
            #TODO argument pos w WordNetLemmatizer ma wplyw na to czy dobrze lematyzuje, np. slowo better musi dostac argument pos="a", zeby zrobic z niego good
            self.tokens = {lemmatizer.lemmatize(w): self.tokens[w] for w in self.tokens}
        else:
            raise ValueError("Document tokens are not correct")

        return self.tokens

    def stemm(self) -> List[str]:
        """
        Perform stemming on every token in self.tokens.
        :return: self.tokens:               list of document's tokens
        """

        self.log_debug("Stemming tokens in {}".format(self.filename))
        if self.tokens and isinstance(self.tokens, dict) and all(isinstance(element, str) for element in self.tokens):
            ps = PorterStemmer()
            self.tokens = {ps.stem(w): self.tokens[w] for w in self.tokens}
        else:
            raise ValueError("Document tokens are not correct")

        return self.tokens

    def preprocess_document(self) -> List[str]:
        """
        Preprocess document content by running all text cleaning functions.
        Steps:
        1. Remove non alphanumeric letters
        2. Tokenize
        3. Remove stop words
        4. Lemmatize
        5. Stemm (optionally, could distort words)
        :return:

        """
        self.log_debug("Preprocessing text in {}".format(self.filename))

        if self.content and isinstance(self.content, str):

            # 1. Remove non alphanumeric letters
            self.remove_non_alphanumeric(to_lower=True)

            # 2. Tokenize
            self.tokenize()

            # 3. Remove stop words
            self.remove_stop_words()

            # 4. Lemmatize
            self.lemmatize()

            # 5. Stemm
            # self.stemm()

            # sort tokens alphabetically
            self.tokens = dict(sorted(self.tokens.items()))

        else:
            raise ValueError("Document content is not correct")

        return self.tokens


if __name__ == "__main__":
    d = Document()
    print(d)
    d.read_content(filename="Computer_Science21.txt")
    print(d)
    d.filename = "bgadflb"
    print(d.id)
    print(d.content)
    print("================================")
    d.preprocess_document()
    print("================================")
    print("content:", d.content)
    print("tokens:", d.tokens)
    print(d)
    pass
