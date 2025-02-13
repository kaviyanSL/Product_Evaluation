import logging
from src.services.LanguageDetectionService import LanguageDetectionService
from src.multiprocess_service.Batching import Batching
from src.database.RawCommentRepository import RawCommentRepository
from multiprocessing import Pool
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Top-level function for language detection
def detect_language(comment):
    language_detection_service = LanguageDetectionService()
    return language_detection_service.detect_language(comment)

class MultiprocessPreprocessText:
    def __init__(self):
        self.Batching = Batching()

    def update_language(self, comment_id, language):
        RawCommentRepository().updating_language(comment_id, language)

    def multiprocess_language_detection(self):
        all_data, number_of_record_per_patch = self.Batching.Batchsize()
        batch_size = 0
        while batch_size < all_data:
            logging.debug(f"Processing batch from {batch_size} to {batch_size + number_of_record_per_patch}")
            batchsized_data = self.Batching.call_batchsized_data(batch_size, batch_size + number_of_record_per_patch)
            batchsized_data = pd.DataFrame(batchsized_data)
            with Pool(processes=os.cpu_count() - 4) as p:
                languages = p.map(detect_language, batchsized_data['comment'])
                for idx, language in enumerate(languages):
                    self.update_language(batchsized_data.iloc[idx]['id'], language)
                    logging.debug(f"Updated language for comment ID {batchsized_data.iloc[idx]['id']} to {language}")

            batch_size += number_of_record_per_patch