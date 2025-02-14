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
import logging
import pandas as pd

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

@blueprint.route("/api/v1/saving_pre_processed_comment/", methods=['POST'])
def saving_pre_processed_comment():
    try:
        db = RawCommentRepository()
        text = db.get_all_raw_comments()
        text = [comment[0] for comment in text]
        pre_processed_text = TextPreProcessorService(text)
        reviews = pre_processed_text.lemmatize()
        db = PreProcessCommentsrepository()
        for pre in reviews:
            db.saving_pre_processed_comments(pre)

        if not reviews:
            return jsonify({"message": "No reviews found on database"}), 200

        return jsonify({"preprocess_reviews": reviews}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@blueprint.route("/api/v1/saving_clustered_comment/", methods=['POST'])
def saving_clustered_comment():
    try:
        db = PreProcessCommentsrepository()
        logging.debug("start for getting all pre processed comments")
        reviews = db.get_all_pre_processed_comments()
        logging.debug("end for getting all pre processed comments")
        reviews = pd.DataFrame(reviews, columns=['id', 'comment'])
        vectorizer = NLPBasedModelsService(reviews['comment'])
        logging.debug("start for Vectorizing reviews")
        vectorize_review = vectorizer.vectorize_reviews()
        logging.debug("end for Vectorizing reviews")
        logging.debug("start for Clustering reviews")
        clustering = ClusteringService(reviews, vectorize_review)
        logging.debug("end for Clustering reviews")
        cluster_data, vectorized_reviews = clustering.get_clustered_reviews()
        db = ClusteredCommentRepository()

        # Collect comments and their clusters for bulk insert
        bulk_insert_data = []
        for cluster in cluster_data.keys():
            if len(cluster_data[cluster]) > 0:
                for comment, vec_comment in zip(cluster_data[cluster], vectorized_reviews[cluster]):
                    bulk_insert_data.append({
                        'comment': comment,
                        'cluster': cluster,
                        'vectorized_comment': vec_comment
                    })

        if bulk_insert_data:
            db.save_clustered_comments(bulk_insert_data)

        if not reviews:
            return jsonify({"message": "No preprocess reviews found in database"}), 200

        return jsonify({"clustered_reviews": cluster_data}), 200
    except Exception as e:
        logging.error("Error during clustering", exc_info=True)
        return jsonify({"error": str(e)}), 500

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
        result = pd.DataFrame(result, columns=['id', 'comment', 'cluster', 'vectorized_comment'])
        clf = ClassificationModelService()
        model_pickle = clf.MLP_Classifier(result['vectorized_comment'], result['cluster'])
        db_classification = ClassificationModelRepository()
        db_classification.saving_classification_model(model_pickle)
        return jsonify({"classification model is created"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500