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
import pandas as pd
blueprint = Blueprint('product_eval',__name__)


@blueprint.route("/api/v1/saving_raw_comment/", methods = ['POST'])
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
            db.saving_raw_comments(raw,lang)
        
        if not text:
            return jsonify({"message": "No reviews found"}), 404

        return jsonify({"reviews": text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@blueprint.route("/api/v1/saving_pre_processed_comment/", methods = ['POST'])
def saving_pre_processed_comment():
    try:
        
        db = RawCommentRepository()
        text = db.get_all_raw_comments()
        text = [comment[0]for comment in text]
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
    


@blueprint.route("/api/v1/saving_clustered_comment/", methods = ['POST'])
def saving_clustered_comment():
    try:
        db = PreProcessCommentsrepository()
        reviews = db.get_all_pre_processed_comments()
        reviews = [comment[0]for comment in reviews]
        vectorizer = NLPBasedModelsService(reviews)
        vectorize_review = vectorizer.vectorize_reviews()
        clustering = ClusteringService(reviews, vectorize_review)
        cluster_data, vectorized_reviews = clustering.get_clustered_reviews()
        db = ClusteredCommentRepository()
        for cluster in cluster_data.keys():
            if len(cluster_data[cluster]) > 0:
                for comment,vec_comment in zip(cluster_data[cluster],vectorized_reviews[cluster]):
                    db.save_clustered_comments(comment,cluster,vec_comment)
        if not reviews:
            return jsonify({"message": "No preprocess reviews found in database"}), 200

        return jsonify({"clustered_reviews": cluster_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@blueprint.route("/api/v1/language_update_multiprocessor/", methods = ['POST'])
def language_update_multiprocessor():
    try:
        multi_lang = MultiprocessPreprocessText()
        multi_lang.multiprocess_language_detection()
        return jsonify({"updating language is done"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500