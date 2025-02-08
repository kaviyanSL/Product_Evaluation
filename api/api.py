from flask import Blueprint, request, jsonify
from services.TagFinderService import TagFinderService
from services.TextPreProcessorService import TextPreProcessorService
from services.NLPBasedModelsService import NLPBasedModelsService
from services.ClusteringService import ClusteringService
from database.SavingClusteredCommentRepository import SavingClusteredCommentRepository
import pandas as pd
blueprint = Blueprint('product_eval',__name__)

@blueprint.route("/api/v1/saving_clustered_comment/", methods = ['POST'])
def scrape_reviews():
    try:
        data = request.get_json()
        url = data.get("url")
        playwright = data.get("playwright", False)  # Default to False

        if not url:
            return jsonify({"error": "URL is required"}), 400

        scraper = TagFinderService(url, playwright)
        text = scraper.find_reviews()
        pre_processed_text = TextPreProcessorService(text)
        reviews = pre_processed_text.lemmatize()
        vectorizer = NLPBasedModelsService(reviews)
        vectorize_review = vectorizer.vectorize_reviews()
        clustering = ClusteringService(reviews, vectorize_review)
        cluster_data = clustering.get_clustered_reviews()
        db = SavingClusteredCommentRepository()
        for i in cluster_data.keys():
            if len(cluster_data[i]) > 0:
                for j in cluster_data[i]:
                    db.save_clustered_comments(j,i)
        if not reviews:
            return jsonify({"message": "No reviews found"}), 404

        return jsonify({"clustered_reviews": cluster_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500