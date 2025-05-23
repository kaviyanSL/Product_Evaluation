from flask import Blueprint, request, jsonify
from src.services.TagFinderService import TagFinderService
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
from src.services.KeywordExtractionService import KeywordExtractionService
from src.database.UserPromptrepository import UserPromptrepository
from src.services.OptimizedPromptGeneratingService import OptimizedPromptGeneratingService
import logging
import pandas as pd
import numpy as np
import os

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
        text = pd.DataFrame(text, columns=['id','comment','website'])
        # text.set_index('id', inplace=True)

        multi_lang = MultiprocessPreprocessText()
        multi_lang.mutlti_processing_tex_lemmatize(text)
        return jsonify({"updating language is done"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@blueprint.route("/api/v1/saving_clustered_comment_bert_embeding/", methods=['POST'])
def saving_clustered_comment_bert_embeding():
    try:
        db = PreProcessCommentsrepository()
        logging.debug("start for getting all pre processed comments")
        reviews = db.get_all_pre_processed_comments()
        logging.debug("end for getting all pre processed comments")
        reviews = pd.DataFrame(reviews, columns=['id', 'comment','website'])
        batch_size_cluster = len(reviews)
        for start in range(0, len(reviews), batch_size_cluster):
            end = min(start + batch_size_cluster, len(reviews))
            reviews_batch = reviews.iloc[start:end].reset_index(drop=True)  # FIX 1: Reset index

            logging.debug(f"reviews DataFrame: {reviews_batch.head()}")

            vectorizer = NLPBasedModelsService(reviews_batch['comment'], reviews_batch['website'].iat[0])  # FIX 2: Use .iat[]
            logging.debug("start for Vectorizing reviews")
            vectorize_review = vectorizer.bert_embedding(reviews_batch['comment'].to_list())
            logging.debug("end for Vectorizing reviews")

            logging.debug("start for Clustering reviews")
            clustering = ClusteringService(reviews_batch['comment'].to_list(), vectorize_review)
            logging.debug("end for Clustering reviews")

            cluster_data, vectorized_reviews = clustering.get_clustered_reviews()

            db = ClusteredCommentRepository()
            bulk_insert_data = []

            for cluster in cluster_data.keys():
                if len(cluster_data[cluster]) > 0:
                    for comment, vec_comment in zip(cluster_data[cluster], vectorized_reviews[cluster]):
                        bulk_insert_data.append({
                            'comment': comment,
                            'cluster': cluster,
                            'vectorize_comment': vec_comment,
                            'website': reviews_batch['website'].iat[0]  # FIX 2: Use .iat[]
                        })

            logging.info(f"bulk_insert_data completed")

            batch_size = 1000
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
    
    
@blueprint.route("/api/v1/creating_BERT_classification_model/", methods=['POST'])
def creating_BERT_classification_models():
    try:
        logging.info(f"ty connecting to db")
        db_cluster = ClusteredCommentRepository()
        result = db_cluster.get_all_clustered_comments()


        logging.info(f"data is red from db")
        result = pd.DataFrame(result, columns=['id', 'comment', 'cluster', 'insert_date', 'vectorized_comment', 'website'])

        result = result.sample(n=100,random_state=42)

        train_data = result.sample(frac=0.95, random_state=42) 
        test_data = result.drop(train_data.index)  
        logging.info(f"dataframe created")

        # Save the 5% data to CSV
        test_data.to_csv("./test_dataset/remaining_5_percent.csv", index=False)
        logging.info(f"test dataset is saved")
        result = train_data

        logging.debug(f"DataFrame head: {result.head()}")

        # Initialize classifier
        clf = ClassificationModelService()
        
        # Get model save path
        model_path = "./models/bert_model.pth"
        clf.bert_classifier(result['comment'].to_list(), result['cluster'])

        # Check if model file exists
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return jsonify({"error": "Model saving failed"}), 500

        # Open and read model binary
        with open(model_path, "rb") as f:
            model_binary = f.read()

        # Save model to the database
        db_classification = ClassificationModelRepository()
        db_classification.saving_classification_model(
            model_name='BERT_Classifier',
            model_data=model_binary, 
            website=result['website'].iloc[0]
        )

        return jsonify({"message": "Classification model is created successfully"}), 200

    except Exception as e:
        logging.error("Error during model creation", exc_info=True)
        return jsonify({"error": str(e)}), 500


@blueprint.route("/api/v1/comment_classifier_predictor/<int:row_id>", methods=['POST'])
def comment_classifier_predictor(row_id):
    try:
        logging.info(f"Processing prediction for row_id: {row_id}")

        if os.path.exists("./models/bert_model.pth"):
            model_path = "./models/bert_model.pth"
        else:
            db_classification = ClassificationModelRepository()
            model_path = db_classification.get_classification_model(model_name="BERT_Classifier")
            if not model_path:
                return jsonify({"error": "Model not found"}), 404

        # Fetch specific comment data
        db_cluster = ClusteredCommentRepository()
        data = db_cluster.get_specific_comment_data(row_id)
        if not data:
            return jsonify({"error": "Comment not found"}), 404

        # Convert tuple to DataFrame
        df = pd.DataFrame([data], columns=["id", "comment", "cluster", "insert_date", "vectorized_comment","website"])

        comments = df['comment'].tolist()

        # Run prediction
        predictor = ClassifierPredictorService(model_path)
        prediction = predictor.predict(comments)[0]

        logging.info(f"Prediction result: {prediction}")
        return jsonify({"prediction": int(prediction)}), 200

    except KeyError as e:
        logging.error(f"KeyError: {e}", exc_info=True)
        return jsonify({"error": f"KeyError: {str(e)}"}), 400

    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return jsonify({"error": str(e)}), 500


@blueprint.route("/api/v1/agg_comment_per_product/<product_name>", methods=['GET'])
def agg_comment_per_product(product_name):
    try:
        logging.info(f"Processing prediction for product: {product_name}")

        if os.path.exists("./models/bert_model.pth"):
            model_path = "./models/bert_model.pth"
        else:
            db_classification = ClassificationModelRepository()
            model_path = db_classification.get_classification_model(model_name="BERT_Classifier")
            if not model_path:
                return jsonify({"error": "Model not found"}), 404


        #TODO: this should be changed based on the given product. result must be the comment based on product
        db_cluster = ClusteredCommentRepository()
        #TODO: this result limited by 200 comment for testing the service
        data = db_cluster.get_all_clustered_comments_200()
        if not data:
            return jsonify({"error": "Comment not found"}), 404

        df = pd.DataFrame(data, columns=["id", "comment", "cluster", "insert_date", "vectorized_comment", "website"])
        #TODO:remember that clear this sample
        df = df.sample(n=1999, random_state=42) 

        comments = df['comment'].tolist()

        predictor = ClassifierPredictorService(model_path)
        prediction_counts = predictor.predict_list(comments)
        #prediction_counts = {int(k): int(v) for k, v in prediction_counts.items()}
        prediction_counts = {int(k): f"{round(int(v)/len(df)*100,2)}%" for k, v in prediction_counts.items()}

        logging.info(f"Prediction resultfor {product_name}: {prediction_counts}")
        return jsonify({product_name:prediction_counts}), 200


    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

@blueprint.route("/api/v1/prompt_generating_sample/", methods=['POST'])
def prompt_generating_sample():
    try:

        data = request.get_json()
        message = data.get('message')
        username = data.get('username')
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        keyword_extracer = KeywordExtractionService(message)
        keywords = keyword_extracer.extracting_keywords()
        prompt_generator = OptimizedPromptGeneratingService(keywords)
        generated_prompt = prompt_generator.generate_text()
        db = UserPromptrepository()
        db.saving_keywords(message,username,keywords,generated_prompt,ip_address)

        return jsonify({"keywords": keywords, "prompt":generated_prompt}), 200


    except Exception as e:
        logging.error("Error during reading message", exc_info=True)
        return jsonify({"error": str(e)}), 500
