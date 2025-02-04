from flask import Blueprint, request, jsonify
from services.TagFinderService import TagFinderService
blueprint = Blueprint('product_eval',__name__)

@blueprint.route("/api/product_url/", methods = ['POST'])
def target_product_url():
    URL_data = request.get_json()
    tag_finder = TagFinderService(URL_data['url'])
    result = tag_finder.readig_html_component()
    return jsonify({"message": "URL received", "URL": result})