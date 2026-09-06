
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def main():

    # ---------------------------------------------------------
    # LOAD PREPARED DATA
    # ---------------------------------------------------------
    train_path = "tourism_project/data/train.csv"
    test_path = "tourism_project/data/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("=" * 60)
    print("MODEL BUILDING AND EXPERIMENTATION")
    print("=" * 60)

    print("\nTrain Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)

    # ---------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # ---------------------------------------------------------
    target_column = "ProdTaken"

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # ---------------------------------------------------------
    # IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
    # ---------------------------------------------------------
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical Features:", numerical_features)
    print("\nCategorical Features:", categorical_features)

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------
    preprocessor = make_column_transformer(
        (
            StandardScaler(),
            numerical_features
        ),
        (
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    )

    # ---------------------------------------------------------
    # DEFINE XGBOOST MODEL
    # ---------------------------------------------------------
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    pipeline = make_pipeline(
        preprocessor,
        model
    )

    # ---------------------------------------------------------
    # HYPERPARAMETER GRID
    # ---------------------------------------------------------
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__subsample": [0.8, 1.0]
    }

    # ---------------------------------------------------------
    # MLFLOW EXPERIMENT
    # ---------------------------------------------------------
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run():

        # -----------------------------------------------------
        # HYPERPARAMETER TUNING
        # -----------------------------------------------------
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )

        print("\nStarting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("\nBest Parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        print("\nBest Cross-Validation F1 Score:",
              round(grid_search.best_score_, 4))

        # -----------------------------------------------------
        # PREDICTIONS
        # -----------------------------------------------------
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # -----------------------------------------------------
        # EVALUATION METRICS
        # -----------------------------------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # -----------------------------------------------------
        # LOG PARAMETERS AND METRICS TO MLFLOW
        # -----------------------------------------------------
        mlflow.log_params(best_params)

        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # -----------------------------------------------------
        # SAVE BEST MODEL
        # -----------------------------------------------------
        model_path = "tourism_project/deployment/best_model.joblib"

        joblib.dump(best_model, model_path)

        print("\nBest model saved successfully at:")
        print(model_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="tourism_package_model"
        )

        print("\nMLflow parameters, metrics, and model logged.")
        print("Model building completed successfully.")


if __name__ == "__main__":
    main()
