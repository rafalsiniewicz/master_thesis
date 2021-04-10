from document import *
import os
from math import log

class Corpus(Base):
    def __init__(self, name="CORPUS"):
        super().__init__(name)
        self.documents: List[Document] = []
        self.number_of_docs: int = 0
        self.vocabulary: List[str] = []             # list of all unique words in the corpus
        # self.features: List[List[float]] = None     #
        self.df: dict[str: float] = dict()
        self.idf: dict[str: float] = dict()


    def read_documents(self):
        """
        Read documents in corpus.
        """
        for doc in os.listdir():
            if ".txt" in doc:
                d = Document()
                d.read_content(filename=doc)
                d.preprocess_document()
                d.log_info("tokens: {}".format(d.tokens))
                self.documents.append(d)


    def get_vocabulary(self):
        """
        Collect all unique words from corpus documents and keep them in self.vocabulary
        :return: self.vocabulary (list[str]):           corpus vocabulary
        """
        self.log_debug("Building a vocabulary from documents words")
        self.vocabulary = []
        for doc in self.documents:
            for word in doc.tokens:
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
                    self.df[word] = 0

        self.vocabulary.sort()
        return self.vocabulary


    def tf_idf(self):
        # calculate tf(t,d) for each term/word in vocabulary from formula: tf(t,d) = count of t in d / number of words in d
        # df(t) = number of documents in which the term is present
        # idf(t) = log [ n / (df(t) + 1) ]).
        for doc in self.documents:
            for term in self.vocabulary:
                if term in doc.tokens.keys():
                    doc.tf[term] = doc.tokens[term] / sum(doc.tokens.values())
                    self.df[term] += 1
                else:
                    doc.tf[term] = 0

        for doc in self.documents:
            for term in self.vocabulary:
                self.idf[term] = log(len(self.documents) / (self.df[term] + 1))
                doc.tf_idf[term] = doc.tf[term] * self.idf[term]

        for doc in self.documents:
            doc.features = list(doc.tf_idf.values())


if __name__ == "__main__":
    c = Corpus()
    c.read_documents()
    c.get_vocabulary()
    print("vocabulary:\n", c.vocabulary)
    for i in c.documents:
        print(i.tokens)
        # print(i.content)
    c.tf_idf()
    for d in c.documents:
        # print("d.tf: ", d.tf)
        print("d.tf: ", d.tf)
        print("d.tf_idf: ", d.tf_idf)
    # print("df:", c.df)

