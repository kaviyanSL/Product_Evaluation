from flask import Blueprint, request, jsonify
blueprint = Blueprint('product_eval',__name__)

@blueprint.route("/api/product_url/", methods = ['POST'])
def target_product_url():
    data = request.get_json()
    return jsonify({"message": "URL received", "URL": data})