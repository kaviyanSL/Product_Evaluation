from keybert import KeyBERT
from src.services.TextPreProcessorService import TextPreProcessorService
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class KeywordExtraction:
    def __init__(self,comment):
        self.comment = comment

    def extracting_keywords(self):
        preprocessor = TextPreProcessorService()
        processed_comment = preprocessor.preprocess(self.comment)
        logging.debug("preprocessed is done")
        keywords = KeyBERT.extract_keywords(processed_comment)
        logging.debug("keyword is extracted")
        return keywords


