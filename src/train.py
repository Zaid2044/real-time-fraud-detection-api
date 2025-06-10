import pandas as pd
import numpy as np
import os
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score

def train_model():
    """
    This function trains the fraud detection model. It includes:
    1. Loading the data.
    2. Preprocessing (scaling).
    3. Splitting data into train/test sets.
    4. Handling class imbalance using SMOTE.
    5. Training an XGBoost classifier.
    6. Evaluating the model.
    7. Saving the trained model pipeline.
    """
    print("Starting the model training process...")

    # 1. Load Data
    print("Loading data...")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'creditcard.csv')

    df = pd.read_csv(DATA_PATH)

    # 2. Preprocessing
    print("Preprocessing data...")
    # Scale the 'Time' and 'Amount' features
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop the original 'Time' and 'Amount' columns
    df = df.drop(['Time', 'Amount'], axis=1)

    # Define features (X) and target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Train/Test Split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle Class Imbalance with SMOTE
    print("Applying SMOTE to handle class imbalance...")
    # It's crucial to apply SMOTE only on the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Original training set shape: {y_train.value_counts()}")
    print(f"Resampled training set shape: {y_train_resampled.value_counts()}")

    # 5. Train XGBoost Classifier
    print("Training XGBoost model...")
    # The 'scale_pos_weight' is another way to handle imbalance, but SMOTE is often more robust.
    # We will use standard XGBoost params on the SMOTE-balanced data.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # 6. Evaluate the Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Calculate and print AUC-PR and AUC-ROC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    auroc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nArea Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
    print(f"Area Under ROC Curve (AUROC): {auroc:.4f}")
    print("------------------------\n")

    # 7. Save the Model
    print("Saving the trained model...")
    # We save the model itself. The API will handle the scaling part.
    # For a more robust pipeline, one would save the scaler too.
    # We'll keep it simple for now.
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model training process complete. 'model.pkl' saved successfully.")


if __name__ == '__main__':
    train_model()
