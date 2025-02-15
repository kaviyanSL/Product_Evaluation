from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, TFBertModel
import logging

class NLPBasedModelsService():
    def __init__(self, reviews):
        self.reviews = reviews
        self.bert_tokenizer_model = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def vectorize_reviews(self):
        vectorizer = TfidfVectorizer()
        vectorize_reviews = vectorizer.fit_transform(self.reviews)
        logging.debug(f"vectorize_reviews completed")
        return vectorize_reviews
    

    def bert_tokenizer(self, reviews):
        tokenized_reviews = self.bert_tokenizer_model(reviews, padding=True, truncation=True, return_tensors='tf')
        logging.debug(f"bert_tokenizer completed")
        return tokenized_reviews
    
    def bert_embedding(self, reviews):
        tokenized_reviews = self.bert_tokenizer(reviews)
        bert_output = self.bert_model(**tokenized_reviews)
        logging.debug(f"bert_embedding completed")
        return bert_output
        
        