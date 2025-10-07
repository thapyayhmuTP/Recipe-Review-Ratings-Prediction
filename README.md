# Recipe-Review-Ratings-Prediction
Machine-learning project predicting 1–5-star recipe ratings using engagement data and review text, featuring feature engineering, TF-IDF, and classification modeling in Python.


This project applies end-to-end data analytics and machine learning to predict the **star ratings (1–5)** of recipe reviews.  
It uses a dataset of over **18,000 reviews** containing user engagement metrics, categorical metadata, and text feedback.  
All analysis, feature engineering, and modeling were implemented independently by me.

---

## Objective

Predict the **exact star rating** of a review using both structured and text-based features.

---

## Methodology

### 1. Data Cleaning
- Audited the dataset for placeholder “2” values used to encode missing data.  
- Verified target label domain (1–5) and removed invalid entries.  
- Detected zero-inflation in engagement fields (likes/dislikes/responses).  
- Converted timestamps to datetime and profiled outliers.  
- Final cleaned dataset: **16,486 × 24**.

### 2. Exploratory Data Analysis (EDA)
- Identified extreme class imbalance (5★ ≈ 83.9%).  
- Used **Spearman correlation** to handle ordinal targets and outliers.  
- Found strongest negative correlations with `dislikes`, `responses`, and `dislike_index`.  
- Visualized relationships across categories (`region`, `device_type`) and review length.

### 3. Feature Engineering
- Log1p transforms for skewed engagement counts (`responses_log1p`, `user_score_log1p`).  
- Text-derived features:
  - `exclam_per_100w` → exclamation intensity  
  - `complaint_per_100w` → density of negative/complaint terms  
- Frequency encoding for compact categorical representation (`region_freq`, `device_type_freq`).  
- Final modeling table: numeric + text + compact categorical features.

### 4. Modeling
| Model | Key Notes | Macro F1 | Macro AUC |
|--------|------------|----------|-----------|
| Logistic Regression (numeric only) | Balanced weights, 5-fold CV | 0.280 | 0.689 |
| Random Forest | 600 trees, class_weight='balanced' | 0.307 | 0.751 |
| **TF-IDF + Logistic Regression** | text + numeric pipeline | **0.391** | **0.814** |

- **Best Model:** TF-IDF + Logistic — most balanced across all rating classes.  
- Stratified split and class weighting mitigated imbalance; oversampling planned for next iteration.

---

## Key Insights
- **Engagement metrics drive ratings:** more dislikes and responses → lower star ratings.  
- **Text signals matter:** negative language and over-punctuation align with low scores.  
- **Random Forest** achieves high accuracy but over-predicts 5★ ratings.  
- **TF-IDF + Logistic** offers the best trade-off between recall and precision.  

---

## Recommendations
- Monitor **macro-F1** rather than accuracy to prevent majority-class bias.  
- Use model outputs to **triage at-risk (1–3★)** reviews for faster response.  
- Enhance user prompts to encourage detailed, constructive feedback.  
- Mine negative keywords weekly to identify recurring issues.  

---

## Skills Demonstrated
- Data cleaning and preprocessing  
- Feature engineering for numeric and textual data  
- Class-imbalance mitigation  
- Machine-learning pipelines using `scikit-learn`  
- Model evaluation (accuracy, macro-F1, AUC)  
- Insight generation and practical recommendations  

---

## 📂 Files
| Path | Description |
|------|--------------|
| `data/recipe_reviews.csv` | Review dataset provided by the instructor |
| `notebook/recipe_review_ratings_prediction.ipynb` | Full analysis and model implementation |

---

 *Dataset provided as part of coursework.  
Used here strictly for educational and portfolio demonstration purposes.*

