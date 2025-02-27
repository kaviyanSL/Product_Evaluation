from keybert import KeyBERT
from src.services.TextPreProcessorService import TextPreProcessorService
import logging
from textblob import TextBlob
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class KeywordExtractionService:
    def __init__(self, comment):
        self.comment = comment

    def extracting_keywords(self):
        preprocessor = TextPreProcessorService()
        processed_comment = preprocessor.preprocess(self.comment)
        logging.debug("Preprocessing is done")

        # Fix spelling errors before extracting keywords
        processed_comment = str(TextBlob(processed_comment).correct())

        key_model = KeyBERT()
        keywords = key_model.extract_keywords(processed_comment, keyphrase_ngram_range=(1,3),
                                              use_maxsum=True, nr_candidates=20, top_n=10)

        keywords = [keyword[0] for keyword in keywords]
        logging.debug("Keywords extracted before cleaning: %s", keywords)

        # Clean extracted keywords
        keywords = self.clean_keywords(keywords)
        logging.debug("Final cleaned keywords: %s", keywords)

        return keywords

    def clean_keywords(self, keywords):
        # Remove redundant keywords
        unique_keywords = list(set(keywords))  

        # Ensure logical grouping (filter noisy keywords)
        grouped_keywords = [kw for kw in unique_keywords if len(kw.split()) > 1] 
        return grouped_keywords if grouped_keywords else unique_keywords  

