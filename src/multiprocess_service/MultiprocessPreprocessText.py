import logging
from src.services.LanguageDetectionService import LanguageDetectionService
from src.multiprocess_service.Batching import Batching
from src.database.RawCommentRepository import RawCommentRepository
from src.services.TextPreProcessorService import TextPreProcessorService
from src.database.PreProcessCommentsrepository import PreProcessCommentsrepository
from multiprocessing import Pool
import pandas as pd
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
error_log_handler = logging.FileHandler('multiprocess_error.log')
error_log_handler.setLevel(logging.ERROR)
error_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_log_handler.setFormatter(error_log_formatter)
logging.getLogger().addHandler(error_log_handler)

# Top-level function for language detection
def detect_language(comment):
    language_detection_service = LanguageDetectionService()
    return language_detection_service.detect_language(comment)

class MultiprocessPreprocessText:
    def __init__(self):
        self.Batching = Batching()

    def update_language(self, updates):
        if not updates:
            return
        raw_comment_repo = RawCommentRepository()
        raw_comment_repo.bulk_update_language(updates)

    def update_lemmatize(self, lemmatize_comment_list):
        if not lemmatize_comment_list:
            return
        lemmatize_comment = PreProcessCommentsrepository()
        lemmatize_comment.bulk_update_lemmatize(lemmatize_comment_list)

    def multiprocess_language_detection(self):
        all_data, number_of_record_per_patch = self.Batching.Batchsize()
        batch_size = 0
        while batch_size < all_data:
            logging.debug(f"Processing batch from {batch_size} to {batch_size + number_of_record_per_patch}")
            batchsized_data = self.Batching.call_batchsized_data(batch_size, batch_size + number_of_record_per_patch)
            batchsized_data = pd.DataFrame(batchsized_data)
            try:
                with Pool(processes=os.cpu_count() - 4) as p:
                    languages = p.map(detect_language, batchsized_data['text'])
                    updates = []
                    for idx, language in enumerate(languages):
                        if not pd.isna(language):
                            updates.append((batchsized_data.iloc[idx]['id'], language))
                            logging.debug(f"Prepared update for comment ID {batchsized_data.iloc[idx]['id']} to {language}")
                    self.update_language(updates)
            except Exception as e:
                logging.error("Error during multiprocessing language detection", exc_info=True)
            batch_size += number_of_record_per_patch

    def mutlti_processing_tex_lemmatize(self, comment_list):
        tex_pre_processor = TextPreProcessorService(comment_list)
        logging.debug(f"starting the mutlti_processing_tex_lemmatize")

        # Calculate batch size
        total_comments = len(comment_list)
        num_cores = os.cpu_count() - 4
        batch_size = total_comments // num_cores 

        for start in range(0, total_comments, batch_size):
            end = min(start + batch_size, total_comments)
            batch = comment_list[start:end]
            logging.debug(f"batch {len(batch)}")

            # Ensure 'id' column exists and set it as the index
            if 'id' not in batch.columns:
                logging.error("The DataFrame does not have an 'id' column.")
                continue  

            batch.set_index('id', inplace=True)

            try:
                with Pool(processes=num_cores) as p:
                    lemmatized_comments = p.map(tex_pre_processor.preprocess, batch['comment'])
                    lemmatize_comment_list = []
                    logging.debug(f"going for enumerate reviews")
                    for idx, lemmatized_comment in enumerate(lemmatized_comments):
                        if not pd.isna(lemmatized_comment):
                            lemmatize_comment_list.append((batch.index[idx], lemmatized_comment))
                            logging.debug(f"Prepared update for comment ID {batch.index[idx]} to {lemmatized_comment}")
                    self.update_lemmatize(lemmatize_comment_list)
            except Exception as e:
                logging.error("Error during multiprocessing text lemmatization", exc_info=True)
