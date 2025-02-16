from flask import Blueprint, request, jsonify
from src.services.TagFinderService import TagFinderService
from src.services.TextPreProcessorService import TextPreProcessorService
from src.services.NLPBasedModelsService import NLPBasedModelsService
from src.services.ClusteringService import ClusteringService
from src.database.ClusteredCommentRepository import ClusteredCommentRepository
from src.database.RawCommentRepository import RawCommentRepository
from src.database.PreProcessCommentsrepository import PreProcessCommentsrepository
from src.services.LanguageDetectionService import LanguageDetectionService
from src.multiprocess_service.MultiprocessPreprocessText import MultiprocessPreprocessText
from src.database.ClassificationModelRepository import ClassificationModelRepository
from src.services.ClassificationModelService import ClassificationModelService
from src.services.ClassifierPredictorService import ClassifierPredictorService
import scipy.sparse
import torch
import re
import logging
import pandas as pd
import ast
import json
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
blueprint = Blueprint('product_eval', __name__)

@blueprint.route("/api/v1/saving_raw_comment/", methods=['POST'])
def scrape_reviews():
    try:
        data = request.get_json()
        url = data.get("url")
        playwright = data.get("playwright", False)  # Default to False

        if not url:
            return jsonify({"error": "URL is required"}), 400

        scraper = TagFinderService(url, playwright)
        text = scraper.find_reviews()
        db = RawCommentRepository()
        lang_detect = LanguageDetectionService()
        for raw in text:
            lang = lang_detect.detect_language(raw)
            db.saving_raw_comments(raw, lang)

        if not text:
            return jsonify({"message": "No reviews found"}), 404

        return jsonify({"reviews": text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @blueprint.route("/api/v1/saving_pre_processed_comment/", methods=['POST'])
# def saving_pre_processed_comment():
#     try:
#         db = RawCommentRepository()
#         text = db.get_all_raw_comments()
#         text = [comment[0] for comment in text]
#         pre_processed_text = TextPreProcessorService(text)
#         reviews = pre_processed_text.lemmatize()
#         db = PreProcessCommentsrepository()
#         for pre in reviews:
#             db.saving_pre_processed_comments(pre)

#         if not reviews:
#             return jsonify({"message": "No reviews found on database"}), 200

#         return jsonify({"preprocess_reviews": reviews}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @blueprint.route("/api/v1/saving_clustered_comment/", methods=['POST'])
# def saving_clustered_comment():
#     try:
#         db = PreProcessCommentsrepository()
#         logging.debug("start for getting all pre processed comments")
#         reviews = db.get_all_pre_processed_comments()
#         logging.debug("end for getting all pre processed comments")
#         reviews = pd.DataFrame(reviews, columns=['id', 'comment'])
#         logging.debug(f"reviews DataFrame: {reviews.head()}")
        
#         vectorizer = NLPBasedModelsService(reviews['comment'])
#         logging.debug("start for Vectorizing reviews")
#         vectorize_review = vectorizer.vectorize_reviews()
#         logging.debug("end for Vectorizing reviews")
        
#         logging.debug("start for Clustering reviews")
#         clustering = ClusteringService(reviews, vectorize_review)
#         logging.debug("end for Clustering reviews")
        
#         cluster_data, vectorized_reviews = clustering.get_clustered_reviews()
#         # logging.debug(f"cluster_data: {cluster_data}")
#         # logging.debug(f"vectorized_reviews: {vectorized_reviews}")
        
#         db = ClusteredCommentRepository()

#         # Collect comments and their clusters for bulk insert
#         bulk_insert_data = []
#         for cluster in cluster_data.keys():
#             if len(cluster_data[cluster]) > 0:
#                 for comment, vec_comment in zip(cluster_data[cluster], vectorized_reviews[cluster]):
#                     bulk_insert_data.append({
#                         'comment': comment,
#                         'cluster': cluster,
#                         'vectorize_comment': vec_comment
#                     })
#         logging.info(f"bulk_insert_data completed")

#         # Batch the inserts
#         batch_size = 1000  # Adjust the batch size as needed
#         for i in range(0, len(bulk_insert_data), batch_size):
#             batch = bulk_insert_data[i:i + batch_size]
#             try:
#                 logging.info(f"try to save clustered comments")
#                 db.save_clustered_comments(batch)
#             except Exception as e:
#                 logging.error("Error during batch insert", exc_info=True)
#                 continue

#         if reviews.empty:
#             return jsonify({"message": "No preprocess reviews found in database"}), 200

#         return jsonify({"clustered_reviews": cluster_data}), 200

#     except KeyError as e:
#         logging.error(f"KeyError during clustering: {e}", exc_info=True)
#         return jsonify({"error": f"KeyError: {str(e)}"}), 500
#     except Exception as e:
#         logging.error("Error during clustering", exc_info=True)
#         return jsonify({"error": str(e)}), 500
    

@blueprint.route("/api/v1/language_update_multiprocessor/", methods=['POST'])
def language_update_multiprocessor():
    try:
        multi_lang = MultiprocessPreprocessText()
        multi_lang.multiprocess_language_detection()
        return jsonify({"updating language is done"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@blueprint.route("/api/v1/update_lemmatize_multiprocessor/", methods=['POST'])
def update_lemmatize_multiprocessor():
    try:
        db = RawCommentRepository()
        text = db.get_all_raw_comments()
        text = pd.DataFrame(text, columns=['id','comment'])
        # text.set_index('id', inplace=True)

        multi_lang = MultiprocessPreprocessText()
        multi_lang.mutlti_processing_tex_lemmatize(text)
        return jsonify({"updating language is done"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@blueprint.route("/api/v1/creating_classification_model/", methods=['POST'])
def creating_classification_models():
    try:
        db_cluster = ClusteredCommentRepository()
        result = db_cluster.get_all_clustered_comments()
        result = pd.DataFrame(result, columns=['id', 'comment', 'cluster', 'insert_date', 'vectorized_comment'])
        
        # Log the first few entries to inspect the data
        logging.debug(f"DataFrame head: {result.head()}")
        
        # Function to parse the string representation of a sparse matrix
        def parse_sparse_matrix(s):
            pattern = re.compile(r'\(0, (\d+)\)\t([\d\.]+)')
            matches = pattern.findall(s)
            indices = [int(match[0]) for match in matches]
            values = [float(match[1]) for match in matches]
            shape = (1, 414316)  # Assuming the shape is known and fixed
            return scipy.sparse.csr_matrix((values, ([0] * len(indices), indices)), shape=shape)
        
        # Convert the 'vectorized_comment' column to a list of sparse matrices
        vectorized_comments = [parse_sparse_matrix(s) for s in result['vectorized_comment']]
        vectorized_comments = scipy.sparse.vstack(vectorized_comments)
        logging.info("scipy.sparse.vstack(vectorized_comments) is done")
        
        clf = ClassificationModelService()
        model_pickle = clf.DNN_Classifier_old(vectorized_comments, result['cluster'])
        
        db_classification = ClassificationModelRepository()
        db_classification.saving_classification_model(model_pickle)
        
        return jsonify({"classification model is created"}), 200
    except Exception as e:
        logging.error("Error during model creation", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@blueprint.route("/api/v1/saving_clustered_comment_bert_embeding/", methods=['POST'])
def saving_clustered_comment_bert_embeding():
    try:
        db = PreProcessCommentsrepository()
        logging.debug("start for getting all pre processed comments")
        reviews = db.get_all_pre_processed_comments()
        logging.debug("end for getting all pre processed comments")
        reviews = pd.DataFrame(reviews, columns=['id', 'comment'])
        batch_size_cluster = 1000
        for start in range(0,len(reviews),batch_size_cluster):
            end = min(start+batch_size_cluster,len(reviews))
            reviews = reviews.iloc[start:end]
            logging.debug(f"reviews DataFrame: {reviews.head()}")
            
            vectorizer = NLPBasedModelsService(reviews['comment'])
            logging.debug("start for Vectorizing reviews")
            vectorize_review = vectorizer.bert_embedding(reviews['comment'].to_list())
            logging.debug("end for Vectorizing reviews")
            
            logging.debug("start for Clustering reviews")
            clustering = ClusteringService(reviews['comment'].to_list(), vectorize_review)
            logging.debug("end for Clustering reviews")
            
            cluster_data, vectorized_reviews = clustering.get_clustered_reviews()
            # logging.debug(f"cluster_data: {cluster_data}")
            # logging.debug(f"vectorized_reviews: {vectorized_reviews}")
            
            db = ClusteredCommentRepository()

            # Collect comments and their clusters for bulk insert
            bulk_insert_data = []
            for cluster in cluster_data.keys():
                if len(cluster_data[cluster]) > 0:
                    for comment, vec_comment in zip(cluster_data[cluster], vectorized_reviews[cluster]):
                        bulk_insert_data.append({
                            'comment': comment,
                            'cluster': cluster,
                            'vectorize_comment': vec_comment
                        })
            logging.info(f"bulk_insert_data completed")

            # Batch the inserts
            batch_size = 1000  # Adjust the batch size as needed
            for i in range(0, len(bulk_insert_data), batch_size):
                batch = bulk_insert_data[i:i + batch_size]
                try:
                    logging.info(f"try to save clustered comments")
                    db.save_clustered_comments(batch)
                except Exception as e:
                    logging.error("Error during batch insert", exc_info=True)
                    continue
        if reviews.empty:
            return jsonify({"message": "No preprocess reviews found in database"}), 200

        return jsonify({"clustered_reviewsis done"}), 200

    except KeyError as e:
        logging.error(f"KeyError during clustering: {e}", exc_info=True)
        return jsonify({"error": f"KeyError: {str(e)}"}), 500
    except Exception as e:
        logging.error("Error during clustering", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    
@blueprint.route("/api/v1/creating_mlp_classification_model/", methods=['POST'])
def creating_mlp_classification_models():
    try:
        db_cluster = ClusteredCommentRepository()
        result = db_cluster.get_all_clustered_comments()
        result = pd.DataFrame(result, columns=['id', 'comment', 'cluster', 'insert_date', 'vectorized_comment'])
        
        # Log the first few entries to inspect the data
        logging.debug(f"DataFrame head: {result.head()}")

        # Ensure vectorized_comments are in the correct format
        vectorized_comments = [scipy.sparse.csr_matrix(np.fromstring(vc.strip('[]'), sep=' ')) if isinstance(vc, str) else vc for vc in result['vectorized_comment']]
        vectorized_comments = scipy.sparse.vstack(vectorized_comments)
        logging.info("scipy.sparse.vstack(vectorized_comments) is done")
        
        clf = ClassificationModelService()
        model_pickle = clf.dnn_classifier(vectorized_comments, result['cluster'])
        
        db_classification = ClassificationModelRepository()
        db_classification.saving_classification_model(model_name = 'DNN_Classifier',model_pickle = model_pickle)
        
        return jsonify({"classification model is created"}), 200
    except Exception as e:
        logging.error("Error during model creation", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@blueprint.route("/api/v1/creating_BERT_classification_model/", methods=['POST'])
def creating_BERT_classification_models():
    try:
        db_cluster = ClusteredCommentRepository()
        result = db_cluster.get_all_clustered_comments()
        result = pd.DataFrame(result, columns=['id', 'comment', 'cluster', 'insert_date', 'vectorized_comment'])
        
        # Log the first few entries to inspect the data
        logging.debug(f"DataFrame head: {result.head()}")

        # Ensure vectorized_comments are in the correct format
        # vectorized_comments = [scipy.sparse.csr_matrix(np.fromstring(vc.strip('[]'), sep=' ')) if isinstance(vc, str) else vc for vc in result['vectorized_comment']]
        # vectorized_comments = scipy.sparse.vstack(vectorized_comments)
        # logging.info("scipy.sparse.vstack(vectorized_comments) is done")
        
        # clf = ClassificationModelService()
        # model_pickle = clf.bert_vector_classifier_v2(vectorized_comments, result['cluster'])


        clf = ClassificationModelService()
        clf.bert_vector_classifier_v2(result['comment'].to_list(), result['cluster'])
        
        db_classification = ClassificationModelRepository()
        db_classification.saving_classification_model(model_name = 'BERT_Classifier',model_pickle = "models/bert_model.pth")
        
        return jsonify({"classification model is created"}), 200
    except Exception as e:
        logging.error("Error during model creation", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

@blueprint.route("/api/v1/comment_classifier_predictor/<row_id>", methods=['POST'])
def comment_classifier_predictor(row_id):
    try:
        row_id = int(row_id)
        db_classification = ClassificationModelRepository()
        db_classification.get_classification_model(model_name='BERT_Classifier')

        # Extract the bytes data from the Row object
        ######TODO: have to send the comment data from rest but now, calling from database###
        db_cluster = ClusteredCommentRepository()
        data = db_cluster.get_specific_comment_data(row_id)
        
        # Wrap the single row tuple in a list before creating the DataFrame
        data = pd.DataFrame([data], columns=['id', 'comment', 'cluster', 'insert_date', 'vectorized_comment'])
        
        # Ensure all vectorized comments have the same shape
        max_length = max(len(np.fromstring(vc.strip('[]'), sep=' ')) for vc in data['vectorized_comment'])
        vectorized_comments = [scipy.sparse.csr_matrix(np.pad(np.fromstring(vc.strip('[]'), sep=' '), 
                               (0, max_length - len(np.fromstring(vc.strip('[]'), sep=' '))))) if 
                               isinstance(vc, str) else vc for vc in data['vectorized_comment']]
        vectorized_comments = scipy.sparse.vstack(vectorized_comments)

        predictor = ClassifierPredictorService()
        predict = int(np.argmax(predictor.predict(model_pickle, vectorized_comments)))

        return jsonify({"prediction": predict}), 200
    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return jsonify({"error": str(e)}), 500




