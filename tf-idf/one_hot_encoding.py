class OneHotEncoding:
    def __init__(self, data):
        self.data = data
        self._corpus = self._build_corpus()
        self._vocabulary = self._build_vocabulary()
        self._word_to_index = {
            word: index for index, word in enumerate(self._vocabulary)
        }

    def _build_corpus(self):
        corpus = []
        for sentence in self.data:
            corpus.extend(sentence.split())
        return corpus

    def _build_vocabulary(self):
        # Keep insertion order so indices are stable and predictable.
        seen = set()
        vocabulary = []
        for word in self._corpus:
            if word not in seen:
                seen.add(word)
                vocabulary.append(word)
        return vocabulary

    def corpus(self):
        return self._corpus.copy()

    def vocabulary(self):
        return self._vocabulary.copy()

    def one_hot_vector(self, word):
        vector = [0] * len(self._vocabulary)
        index = self._word_to_index.get(word)
        if index is not None:
            vector[index] = 1
        return vector

    def encode_sentence(self, sentence):
        return [self.one_hot_vector(word) for word in sentence.split()]

    def multi_hot_vector(self, sentence):
        vector = [0] * len(self._vocabulary)
        for word in sentence.split():
            index = self._word_to_index.get(word)
            if index is not None:
                vector[index] = 1
        return vector


if __name__ == "__main__":
    data = [
        "people watch youtube",
        "youtube watch people",
        "people comment on youtube",
        "youtube comment on people",
    ]

    encoder = OneHotEncoding(data)
    print("Corpus:", encoder.corpus())
    print("Vocabulary:", encoder.vocabulary())

    sentence = data[0]
    print(
        f"One-hot vectors for words in '{sentence}': "
        f"{encoder.encode_sentence(sentence)}"
    )
    print(f"Multi-hot vector for '{sentence}': {encoder.multi_hot_vector(sentence)}")
