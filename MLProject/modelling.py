import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment("random_forest_personality")

def load_data():
    train = pd.read_csv('personality_dataset_preprocessing/train_data.csv')
    test = pd.read_csv('personality_dataset_preprocessing/test_data.csv')
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

    params = {
        'n_estimators': 50,
        'max_depth': None,
        'random_state': 42,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    with mlflow.start_run(run_name="RF_best_manual"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, preds, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
        prec_weighted = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)

        # Logging metrics (manual, jangan dihapus)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec_macro)
        mlflow.log_metric("recall_macro", rec_macro)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_weighted", prec_weighted)
        mlflow.log_metric("recall_weighted", rec_weighted)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Save metrics to JSON
        metrics_dict = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted
        }
        with open("metric_info.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact("metric_info.json")
        os.remove("metric_info.json")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        cm_path = "training_confusion_matrix.png"
        plot_confusion_matrix(cm, sorted(y_test.unique()), cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Feature importance
        fi_path = "feature_importance.png"
        plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path)
        os.remove(fi_path)

        # Save model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Training finished. Metrics and artifacts logged to MLflow.")

if __name__ == "__main__":
    train_and_log()