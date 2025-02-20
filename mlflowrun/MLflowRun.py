import mlflow
import configparser

# Read config
config = configparser.ConfigParser()
config.read("./mlflowrun/mlflow_config.conf")

class MLflowRun:
    def __init__(self):
        self.set_tracking_uri = mlflow.set_tracking_uri(f"http://{config['mlflow']['IP']}:{config['mlflow']['port']}")
        self.set_experiment = mlflow.set_experiment(config['mlflow']['experiment_name'])

    def start_run(self):
        mlflow.start_run()

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_params(self, params_dict):
        for key, value in params_dict.items():
            mlflow.log_param(key, value)

    def log_artifact(self, file_path, artifact_path=f".{config['mlflow']['artifacts_path']}artifacts"):
        artifact_path = artifact_path.replace("\"","")
        mlflow.log_artifact(file_path, artifact_path)

    def log_text(self, text, artifact_file):
        artifact_file_path = f".{config['mlflow']['text_path']}{artifact_file}"
        artifact_file_path = artifact_file_path.replace("\"","")
        with open(artifact_file_path, "w") as f:
            f.write(text)
        mlflow.log_artifact(artifact_file_path)

    def end_run(self):
        mlflow.end_run()
