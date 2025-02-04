from flask import Flask
from api.api import blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    return app

if __name__ == "__manin__":
    app = create_app()
    app.run(debug=True)