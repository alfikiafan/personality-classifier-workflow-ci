import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
import platform
import sys
import shutil

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
exp = mlflow.get_experiment_by_name("random_forest_personality")
if exp is None or exp.lifecycle_stage == 'deleted':
    exp_id = mlflow.create_experiment("cc")
else:
    exp_id = exp.experiment_id
mlflow.set_experiment(exp_id)
print(f"Using experiment ID: {exp_id}")

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'personality_dataset_preprocessing', 'train_data.csv')
    test_path = os.path.join(base_dir, 'personality_dataset_preprocessing', 'test_data.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop('Personality', axis=1)
    y_train = train['Personality']
    X_test = test.drop('Personality', axis=1)
    y_test = test['Personality']
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(cm, classes, output_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    sns.barplot(x=[importances[i] for i in indices], y=[feature_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_and_log():
    X_train, y_train, X_test, y_test = load_data()
    feature_names = X_train.columns.tolist()

    os.makedirs("artifacts", exist_ok=True)

    params = {
        'n_estimators': 50,
        'max_depth': None,
        'random_state': 42,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    with mlflow.start_run(run_name="RF_best_manual") as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, preds, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec_macro)
        mlflow.log_metric("recall_macro", rec_macro)
        mlflow.log_metric("f1_macro", f1_macro)

        metrics_dict = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro
        }
        with open("artifacts/metric_info.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact("artifacts/metric_info.json")

        cm = confusion_matrix(y_test, preds)
        cm_path = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(cm, sorted(y_test.unique()), cm_path)
        mlflow.log_artifact(cm_path)

        if len(model.classes_) == 2:
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test.map({model.classes_[0]: 0, model.classes_[1]: 1}), probs)
            roc_auc = roc_auc_score(y_test.map({model.classes_[0]: 0, model.classes_[1]: 1}), probs)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = "artifacts/roc_curve.png"
            plt.savefig(roc_path)
            plt.close()

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_artifact(roc_path)

        fi_path = "artifacts/feature_importance.png"
        plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path)

        cls_report = classification_report(y_test, preds, output_dict=True)
        with open("artifacts/classification_report.json", "w") as f:
            json.dump(cls_report, f, indent=4)
        mlflow.log_artifact("artifacts/classification_report.json")

        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        pred_df.to_csv("artifacts/predictions.csv", index=False)
        mlflow.log_artifact("artifacts/predictions.csv")

        joblib.dump(model, "artifacts/model.pkl")
        mlflow.log_artifact("artifacts/model.pkl")

        mlflow.sklearn.log_model(model, "model")

        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "mlflow_version": mlflow.__version__,
            "pandas_version": pd.__version__,
        }
        with open("artifacts/environment_info.json", "w") as f:
            json.dump(env_info, f, indent=4)
        mlflow.log_artifact("artifacts/environment_info.json")

        print("Training completed. All metrics and artifacts are logged.")

        local_download_dir = "downloaded_artifacts"
        shutil.rmtree(local_download_dir, ignore_errors=True)
        os.makedirs(local_download_dir, exist_ok=True)

        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            client.download_artifacts(run.info.run_id, artifact.path, local_download_dir)
        print(f"All artifacts downloaded to: {local_download_dir}")

if __name__ == "__main__":
    train_and_log()