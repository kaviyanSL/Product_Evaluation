from src.services.LanguageDetectionService import LanguageDetectionService
from src.services.ClusteringService import ClusteringService
from src.services.NLPBasedModelsService import NLPBasedModelsService
from src.services.TextPreProcessorService import TextPreProcessorService
from src.multiprocess_service.Batching import Batching
from src.database.RawCommentRepository import RawCommentRepository
from multiprocessing import Pool
import pandas as pd
import os

class MultiprocessPreprocessText:
    def __init__(self):
        self.LanguageDetectionService = LanguageDetectionService()
        self.ClusteringService = ClusteringService()
        self.NLPBasedModelsService = NLPBasedModelsService()
        self.TextPreProcessorService = TextPreProcessorService()
        self.Batching = Batching()
        self.RawCommentRepository = RawCommentRepository()

    def detect_language(self, comment):
        return self.LanguageDetectionService.detect_language(comment)

    def multiprocess_language_detection(self):
        all_data, number_of_record_per_patch = self.Batching.Batchsize()
        batch_size = 0
        while batch_size < all_data:
            batchsized_data = self.Batching.call_batchsized_data(batch_size, batch_size + number_of_record_per_patch)
            batchsized_data = pd.DataFrame(batchsized_data)
            with Pool(processes=os.cpu_count() - 4) as p:
                languages = p.map(self.detect_language, batchsized_data['comment'])
                for idx, language in enumerate(languages):
                    self.RawCommentRepository.updating_language(batchsized_data.iloc[idx]['id'], language)

            batch_size += number_of_record_per_patch