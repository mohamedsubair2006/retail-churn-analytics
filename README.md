# 🛒 Retail Intelligence & Customer Churn Prediction System

An end-to-end data analytics and machine learning system that aggregates transactional e-commerce data, predicts customer churn risk using XGBoost, and exposes actionable business metrics through an interactive Gradio dashboard.

## 🚀 Key Highlights & Performance

* **ETL Pipeline (`etl_pipeline.py`)**: Processed raw e-commerce transaction logs using **DuckDB** and **SQL CTEs** to aggregate Recency, Frequency, and Monetary (RFM) metrics across 500,000+ records.
* **ML Machine Learning (`train_model.py`)**: Built an **XGBoost Classifier** to predict churn risk while eliminating target data leakage and handling class imbalance via `scale_pos_weight`.
  * **Model Recall**: 89% (High sensitivity for identifying churners)
  * **ROC-AUC Score**: 0.9532
* **Interactive Dashboard (`gradio_app.py`)**: Built a responsive UI featuring risk-tier segmentation, lifetime spend distribution plots, and a **What-If Retention Simulator** to quantify revenue at risk.

## 🛠️ Tech Stack

* **Language**: Python 3.13
* **Data Processing**: Pandas, DuckDB, SQL
* **Machine Learning**: XGBoost, Scikit-Learn
* **Visualization & UI**: Gradio, Plotly

## 💻 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/mohamedsubair2006/retail-churn-analytics.git](https://github.com/mohamedsubair2006/retail-churn-analytics.git)
   cd retail-churn-analytics