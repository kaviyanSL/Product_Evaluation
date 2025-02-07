from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer

class TextPreProcessorService():
    def __init__(self,comment_list):
        self.comment_list = comment_list
    
    def lower_case(self):
        try:
            list_of_lowercase_comment = [comment.lower() for comment in self.comment_list]
            return list_of_lowercase_comment
        except Exception as e:
            return str(e)
        
    def remove_punctuation(self):
        try:
            lower_case_comments = self.lower_case()
            removed_punctuation = [re.sub(r'[^a-zA-Z\s]', '', i) for i in lower_case_comments]
            return removed_punctuation
        except Exception as e:
            return str(e)
        
    def remove_stopwords(self):
        try:
            removed_punctuation = self.remove_punctuation()
            stopwords = set(stopwords.words('english'))
            removed_stopwords = [' '.join([word for word in comment.split() if word not in stopwords])
                                  for comment in removed_punctuation]
            return removed_stopwords
        except Exception as e:
            return str(e)
    
    def lemmatize(self):
        try:
            removed_stopwords = self.remove_stopwords()
            lemmatizer = WordNetLemmatizer()
            lemmatized_comments = [' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
                                   for comment in removed_stopwords]
            return lemmatized_comments
        except Exception as e:
            return str(e)
    
        
    
    