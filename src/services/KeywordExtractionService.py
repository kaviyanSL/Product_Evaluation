from keybert import KeyBERT
from src.services.TextPreProcessorService import TextPreProcessorService
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class KeywordExtractionService:
    def __init__(self,comment):
        self.comment = comment

    def extracting_keywords(self):
        preprocessor = TextPreProcessorService()
        processed_comment = preprocessor.preprocess(self.comment)
        logging.debug("preprocessed is done")
        key_model = KeyBERT()
        keywords = key_model.extract_keywords(processed_comment, keyphrase_ngram_range=(1,4),
                                              use_maxsum=True, nr_candidates=20, top_n=10)
        keywords = [keyword[0] for keyword in keywords]
        logging.debug("keyword is extracted")
        return keywords


