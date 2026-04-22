# ✈️ Airline Customer Sentiment AI Analysis
![Accuracy Score](https://img.shields.io/badge/Accuracy-97.25%25-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)

## 📊 Project Overview
This project is an end-to-end Data Science pipeline that analyzes **6,000+ customer reviews** to predict whether a passenger will recommend an airline. By combining numerical ratings with Natural Language Processing (NLP), we achieved a peak prediction accuracy of **97.25%**.

## 🚀 Key Technical Pillars
*   **NLP Engine:** Uses **TF-IDF Vectorization** and **VADER Sentiment Analysis** to translate human emotions into data.
*   **Predictive AI:** Implements **Random Forest** and **XGBoost** with automated hyperparameter tuning.
*   **Business Intelligence:** Generates 15+ analytical visualizations to identify the root causes of passenger dissatisfaction.
*   **Modular Pipeline:** Built using **Scikit-Learn Pipelines** for professional-grade code maintainability.

## 📈 Key Discoveries
*   **Primary Driver:** "Value for Money" is the strongest indicator of a recommendation.
*   **Staff Performance:** Cabin staff service is consistently high, even in poor reviews, suggesting that service failures are primarily **operational** (delays, aircraft age).
*   **Model Performance:** The Combined Model (Numerical + Text) outperformed standalone models by ~5%.

## 📂 Project Structure
- `run_project_v2.py`: The main AI pipeline (Modern Tech version).
- `analytics_v2_results/`: Folder containing all generated charts and CSV reports.
- `PROJECT_OVERVIEW.txt`: A simplified text summary for quick reading.
- `MGT0000_...xlsx`: The dataset used for analysis.

## 🛠️ How to Run
1. Install dependencies: `pip install pandas seaborn scikit-learn xgboost afinn wordcloud`
2. Run the pipeline: `python run_project_v2.py`

---
*Developed by **meetp** in collaboration with Antigravity AI.*
