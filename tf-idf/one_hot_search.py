from one_hot_encoding import OneHotEncoding


class OneHotSearch(OneHotEncoding):
    def cosine_similarity(self, sentence_a, sentence_b):
        vector_a = self.multi_hot_vector(sentence_a)
        vector_b = self.multi_hot_vector(sentence_b)

        dot_product = 0
        for value_a, value_b in zip(vector_a, vector_b):
            dot_product += value_a * value_b

        magnitude_a = sum(value * value for value in vector_a) ** 0.5
        magnitude_b = sum(value * value for value in vector_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def search(self, query, top_k=3):
        results = []

        for sentence in self.data:
            score = self.cosine_similarity(query, sentence)
            results.append((sentence, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def recommend(self, liked_sentence, top_k=2):
        results = []

        for sentence in self.data:
            if sentence == liked_sentence:
                continue

            score = self.cosine_similarity(liked_sentence, sentence)
            results.append((sentence, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def print_results(results):
    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"{rank}. {sentence} -> similarity score: {score:.3f}")


def get_operation():
    print("Choose an operation:")
    print("1. Search")
    print("2. Recommend")
    choice = input("Enter 1 or 2: ").strip().lower()

    if choice in {"1", "search", "s"}:
        return "search"
    if choice in {"2", "recommend", "recommendation", "r"}:
        return "recommend"
    return None


def get_liked_item(search_engine):
    print("Available items:")
    for index, sentence in enumerate(search_engine.data, start=1):
        print(f"{index}. {sentence}")

    user_input = input(
        "Enter an item number or describe what you like: "
    ).strip()

    if user_input.isdigit():
        selected_index = int(user_input) - 1
        if 0 <= selected_index < len(search_engine.data):
            return search_engine.data[selected_index]
        return None

    closest_match = search_engine.search(user_input, top_k=1)
    if not closest_match or closest_match[0][1] == 0:
        return None

    liked_sentence = closest_match[0][0]
    print(f"Using the closest match: {liked_sentence}")
    return liked_sentence


if __name__ == "__main__":
    data = [
        "python basics for beginners",
        "advanced python for data analysis",
        "javascript for web development",
        "machine learning with python",
        "data science projects using python",
    ]

    search_engine = OneHotSearch(data)
    print("Vocabulary:", search_engine.vocabulary())
    print()

    while True:
        operation = get_operation()
        print()

        if operation == "search":
            query = input("Enter your search query: ").strip()
            print()
            print("Search results:")
            print_results(search_engine.search(query))
        elif operation == "recommend":
            liked_course = get_liked_item(search_engine)
            print()

            if liked_course is None:
                print("Could not find a matching item for recommendation.")
            else:
                print("Recommendations based on:", liked_course)
                print_results(search_engine.recommend(liked_course))
        else:
            print("Invalid choice. Please select search or recommend.")

        print()
        continue_choice = input("Do you want to perform another operation? (y/n): ")
        if continue_choice.strip().lower() != "y":
            break
        print()
