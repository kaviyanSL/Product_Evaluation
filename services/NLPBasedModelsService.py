from sklearn.feature_extraction.text import TfidfVectorizer

class NLPBasedModelsService():
    def __init__(self, reviews):
        self.reviews = reviews

    def vectorize_reviews(self):
        vectorizer = TfidfVectorizer()
        vectorize_reviews = vectorizer.fit_transform(self.reviews)
        return vectorize_reviews

