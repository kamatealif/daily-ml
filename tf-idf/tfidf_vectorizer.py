import math


class TFIDFVectorizer:
    def __init__(self, documents):
        self.documents = documents
        self._tokenized_documents = [
            self._tokenize(document) for document in documents
        ]
        self._vocabulary = self._build_vocabulary()
        self._document_frequencies = self._build_document_frequencies()
        self._idf_scores = self._build_idf_scores()

    def _tokenize(self, text):
        tokens = []
        for word in text.split():
            token = word.lower().strip(".,!?;:")
            if token:
                tokens.append(token)
        return tokens

    def _build_vocabulary(self):
        seen = set()
        vocabulary = []

        for tokens in self._tokenized_documents:
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocabulary.append(token)

        return vocabulary

    def _build_document_frequencies(self):
        document_frequencies = {token: 0 for token in self._vocabulary}

        for tokens in self._tokenized_documents:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequencies[token] += 1

        return document_frequencies

    def _build_idf_scores(self):
        total_documents = len(self.documents)
        idf_scores = {}

        for token in self._vocabulary:
            document_frequency = self._document_frequencies[token]
            idf_scores[token] = math.log(
                (1 + total_documents) / (1 + document_frequency)
            ) + 1

        return idf_scores

    def vocabulary(self):
        return self._vocabulary.copy()

    def idf_scores(self):
        return self._idf_scores.copy()

    def vectorize(self, document):
        tokens = self._tokenize(document)
        vector = []

        if not tokens:
            return [0.0] * len(self._vocabulary)

        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        total_tokens = len(tokens)

        for token in self._vocabulary:
            term_frequency = term_counts.get(token, 0) / total_tokens
            tfidf_score = term_frequency * self._idf_scores[token]
            vector.append(tfidf_score)

        return vector

    def encode_corpus(self):
        return [self.vectorize(document) for document in self.documents]

    def top_terms(self, document, top_k=3):
        vector = self.vectorize(document)
        weighted_terms = list(zip(self._vocabulary, vector))
        weighted_terms.sort(key=lambda item: item[1], reverse=True)

        return [
            (term, score) for term, score in weighted_terms[:top_k] if score > 0
        ]


if __name__ == "__main__":
    documents = [
        "Python is great for machine learning projects.",
        "Machine learning projects often use Python and data.",
        "Data science and Python work well together.",
        "Football and cricket are popular sports.",
    ]

    vectorizer = TFIDFVectorizer(documents)

    print("Vocabulary:", vectorizer.vocabulary())
    print()

    print("IDF scores:")
    for token, score in vectorizer.idf_scores().items():
        print(f"{token}: {score:.3f}")

    print()

    for index, document in enumerate(documents, start=1):
        vector = vectorizer.vectorize(document)
        rounded_vector = [round(value, 3) for value in vector]

        print(f"Document {index}: {document}")
        print("TF-IDF vector:", rounded_vector)
        print("Top terms:", vectorizer.top_terms(document))
        print()
