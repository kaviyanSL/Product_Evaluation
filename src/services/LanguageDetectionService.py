
from langdetect import detect_langs
from langdetect import DetectorFactory
import numpy as np

DetectorFactory.seed = 42

class LanguageDetectionService:
    def __init__(self):
        pass

    def detect_language(self, comment):
        try:
            detected_langs = detect_langs(comment)
            if detected_langs:
                max_prob_lang = max(detected_langs, key=lambda x: x.prob)
                return max_prob_lang.lang
            return np.nan
        except Exception as e:
            return np.nan