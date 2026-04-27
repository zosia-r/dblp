from .retriever import Retriever
from .generator import Generator


class RAG:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

        self.retriever.index_if_needed()

    def run(self, query):
        results = self.retriever.search(query)

        docs = results["documents"][0]

        answer = self.generator.generate(query, docs)

        return answer, docs