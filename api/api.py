from flask import Blueprint, request, jsonify
from services.TagFinderService import TagFinderService
from services.TextPreProcessorService import TextPreProcessorService
blueprint = Blueprint('product_eval',__name__)

@blueprint.route("/api/product_url/", methods = ['POST'])
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

        if not reviews:
            return jsonify({"message": "No reviews found"}), 404

        return jsonify({"reviews": reviews}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500