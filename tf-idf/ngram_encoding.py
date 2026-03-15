class NGramEncoding:
    def __init__(self, data, n=2):
        if n < 1:
            raise ValueError("n must be greater than or equal to 1")

        self.data = data
        self.n = n
        self._corpus = self._build_corpus()
        self._vocabulary = self._build_vocabulary()
        self._ngram_to_index = {
            ngram: index for index, ngram in enumerate(self._vocabulary)
        }

    def _tokenize(self, sentence):
        tokens = []
        for word in sentence.split():
            tokens.append(word.lower().strip(".,!?;:"))
        return tokens

    def _generate_ngrams(self, tokens):
        if len(tokens) < self.n:
            return []

        ngrams = []
        for index in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[index : index + self.n])
            ngrams.append(ngram)
        return ngrams

    def _build_corpus(self):
        corpus = []
        for sentence in self.data:
            tokens = self._tokenize(sentence)
            corpus.extend(self._generate_ngrams(tokens))
        return corpus

    def _build_vocabulary(self):
        # Keep insertion order so vectors stay stable across runs.
        seen = set()
        vocabulary = []
        for ngram in self._corpus:
            if ngram not in seen:
                seen.add(ngram)
                vocabulary.append(ngram)
        return vocabulary

    def corpus(self):
        return [" ".join(ngram) for ngram in self._corpus]

    def vocabulary(self):
        return [" ".join(ngram) for ngram in self._vocabulary]

    def vectorize(self, sentence):
        vector = [0] * len(self._vocabulary)
        tokens = self._tokenize(sentence)

        for ngram in self._generate_ngrams(tokens):
            index = self._ngram_to_index.get(ngram)
            if index is not None:
                vector[index] += 1

        return vector

    def encode_corpus(self):
        return [self.vectorize(sentence) for sentence in self.data]


if __name__ == "__main__":
    data = [
        "people watch youtube videos",
        "people watch youtube shorts",
        "youtube videos help people learn",
        "youtube shorts entertain people",
    ]

    encoder = NGramEncoding(data, n=2)

    print("N value:", encoder.n)
    print("Bi-gram corpus:", encoder.corpus())
    print("Bi-gram vocabulary:", encoder.vocabulary())
    print()

    for index, sentence in enumerate(data, start=1):
        print(f"Sentence {index} bi-gram vector:", encoder.vectorize(sentence))
