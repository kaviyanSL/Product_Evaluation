from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class NLPBasedModelsService():
    def __init__(self, reviews):
        self.reviews = reviews

    def vectorize_reviews(self):
        vectorizer = TfidfVectorizer()
        vectorize_reviews = vectorizer.fit_transform(self.reviews)
        logging.debug(f"vectorize_reviews completed")
        return vectorize_reviews

