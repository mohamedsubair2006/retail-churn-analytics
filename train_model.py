import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def train_churn_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, 'processed_customer_features.csv')
    
    print("Loading processed features...")
    df = pd.read_csv(input_csv)

    # FIX: Exclude 'recency_days' to prevent direct target leakage!
    feature_cols = ['frequency', 'monetary_value', 'avg_basket_size', 'customer_tenure_days']
    X = df[feature_cols]
    y = df['is_churned']

    # Train-Test Split with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Handle Imbalance: Calculate scale_pos_weight
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate Model Performance
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*40)
    print("      REALISTIC MODEL PERFORMANCE EVALUATION      ")
    print("="*40)
    print(classification_report(y_test, y_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    print("="*40 + "\n")

    # Generate Probability & Assign Risk Tiers for Dashboard
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    df['risk_level'] = pd.cut(
        df['churn_probability'], 
        bins=[-0.01, 0.35, 0.70, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Save Artifacts for Streamlit Dashboard
    output_csv = os.path.join(base_dir, 'final_scored_customers.csv')
    model_pkl = os.path.join(base_dir, 'xgb_churn_model.pkl')
    
    df.to_csv(output_csv, index=False)
    with open(model_pkl, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saved predictions: '{output_csv}'")
    print(f"Saved trained model: '{model_pkl}'")

if __name__ == '__main__':
    train_churn_model()