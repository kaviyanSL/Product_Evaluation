import logging
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import nltk
import os
nltk_data_path = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(nltk_data_path)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPreProcessorService():
    def __init__(self, comment_list):
        self.comment_list = comment_list

    def lower_case(self, comment):
        return comment.lower()

    def remove_punctuation(self, comment):
        return re.sub(r'[^a-zA-Z\s]', '', comment)

    def remove_stopwords(self, comment):
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in comment.split() if word not in stop_words])

    def lemmatize(self, comment):
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    def preprocess(self, comment):
        try:
            comment = self.lower_case(comment)
            comment = self.remove_punctuation(comment)
            comment = self.remove_stopwords(comment)
            comment = self.lemmatize(comment)
            logging.info(f"comment is lemmatize {comment[:20]}")
            return comment
        except Exception as e:
            logging.error(e, exc_info=True)
            return str(e)