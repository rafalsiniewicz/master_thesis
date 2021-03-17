from typing import List
from document import Document
import os

class Corpus:
    def __init__(self):
        self.documents: List[Document] = []
        self.number_of_docs: int = 0


    def read_documents(self):
        for doc in os.listdir():
            if ".txt" in doc:
                d = Document()
                d.read_content(filename=doc)
                d.preprocess_document()
                print("tokens:", d.tokens)
                self.documents.append(d)



if __name__ == "__main__":
    c = Corpus()
    c.read_documents()
    for i in c.documents:
        print(i)
