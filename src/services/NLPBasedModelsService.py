from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import logging
from tqdm import tqdm
import time
import numpy as np
import torch
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class NLPBasedModelsService:
    def __init__(self, reviews, website):
        # Set up GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.reviews = reviews
        self.website = website

        # Load BERT tokenizer and model
        self.bert_tokenizer_model = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)  # Move model to GPU

    def vectorize_reviews(self):
        """Vectorize reviews using TF-IDF (runs on CPU)"""
        vectorizer = TfidfVectorizer()
        vectorized_reviews = vectorizer.fit_transform(self.reviews)
        logging.debug("vectorize_reviews completed")
        return vectorized_reviews

    def bert_tokenizer(self, reviews):
        """Tokenize reviews using BERT tokenizer"""
        start_time = time.time()
        tokenized_reviews = []
        
        for review in tqdm(reviews, desc="Tokenizing reviews"):
            tokenized_review = self.bert_tokenizer_model(
                review, padding=True, truncation=True, return_tensors='pt'  # Use PyTorch tensors
            ).to(self.device)  # Move to GPU
            tokenized_reviews.append(tokenized_review)
        
        end_time = time.time()
        logging.debug(f"bert_tokenizer completed in {end_time - start_time:.2f} seconds")
        return tokenized_reviews

    def bert_embedding(self, reviews):
        """Generate BERT embeddings on GPU"""
        start_time = time.time()
        tokenized_reviews = self.bert_tokenizer(reviews)  # Tokenize input

        embeddings = []
        for tokenized_review in tqdm(tokenized_reviews, desc="Generating BERT embeddings"):
            with torch.no_grad():  # No gradient computation (faster inference)
                bert_output = self.bert_model(**tokenized_review)  # Forward pass
                cls_embeddings = bert_output.last_hidden_state[:, 0, :].to('cpu').numpy()  # Move to CPU for numpy
                
            embeddings.append(cls_embeddings)

        embeddings = np.vstack(embeddings)  # Stack all embeddings
        end_time = time.time()
        logging.debug(f"bert_embedding completed in {end_time - start_time:.2f} seconds")
        
        return embeddings
