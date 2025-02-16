from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel
import logging
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf

class NLPBasedModelsService():
    def __init__(self, reviews):
        self.reviews = reviews
        self.bert_tokenizer_model = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logging.info("GPU is available and will be used for BERT embeddings.")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU: {e}")


        # Ensure TensorFlow uses GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logging.info("GPU is available and will be used for BERT embeddings.")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU: {e}")

    def vectorize_reviews(self):
        vectorizer = TfidfVectorizer()
        vectorize_reviews = vectorizer.fit_transform(self.reviews)
        logging.debug(f"vectorize_reviews completed")
        return vectorize_reviews

    def bert_tokenizer(self, reviews):
        start_time = time.time()
        tokenized_reviews = []
        for review in tqdm(reviews, desc="Tokenizing reviews"):
            tokenized_review = self.bert_tokenizer_model(review, padding=True, truncation=True, return_tensors='tf')
            tokenized_reviews.append(tokenized_review)
        end_time = time.time()
        logging.debug(f"bert_tokenizer completed in {end_time - start_time:.2f} seconds")
        return tokenized_reviews

    def bert_embedding(self, reviews):
        start_time = time.time()
        tokenized_reviews = self.bert_tokenizer(reviews)
        embeddings = []
        for tokenized_review in tqdm(tokenized_reviews, desc="Generating BERT embeddings"):
            bert_output = self.bert_model(**tokenized_review)
            # Extract the [CLS] token's embeddings
            cls_embeddings = bert_output.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embeddings)
        embeddings = np.vstack(embeddings)
        end_time = time.time()
        logging.debug(f"bert_embedding completed in {end_time - start_time:.2f} seconds")
        return embeddings