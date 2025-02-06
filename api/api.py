from flask import Blueprint, request, jsonify
from services.TagFinderService import TagFinderService
blueprint = Blueprint('product_eval',__name__)

@blueprint.route("/api/product_url/", methods = ['POST'])
def scrape_reviews():
    try:
        data = request.get_json()
        url = data.get("url")
        use_selenium = data.get("use_selenium", False)  # Default to False

        if not url:
            return jsonify({"error": "URL is required"}), 400

        scraper = TagFinderService(url, use_selenium)
        reviews = scraper.find_reviews()

        if not reviews:
            return jsonify({"message": "No reviews found"}), 404

        return jsonify({"reviews": reviews}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500