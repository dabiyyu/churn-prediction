# Customer Churn Prediction

## Business Problem
A telecom company loses revenue every time a customer churns.
Acquiring a new customer costs 5–7x more than retaining one.
This project identifies high-risk customers before they churn
so the retention team can intervene proactively.

## Key Findings
- **27% overall churn rate** — 1,869 customers lost out of 7,043
- **$7.8M in CLTV lost** to churn
- Month-to-month contracts churn at **42.7%** vs 2.8% for two-year contracts
- Customers in their **first 12 months** are the highest-risk segment
- Churned customers pay **$13 more/month** on average than retained ones
- Top churn reasons: poor support experience + competitor product advantages
- Fiber optic users churn at **41.9%** despite paying more — product gap vs competitors

## Model Performance
| Model | ROC-AUC |
|---|---|
| Gradient Boosting | **0.8501** |
| Logistic Regression | 0.8491 |
| Random Forest | 0.8392 |

Best model: **Gradient Boosting** — correctly identified 198 out of 374 churners on the test set.

## Top Predictive Features (SHAP)
1. Tenure Months — longer tenure = lower churn risk
2. Dependents — family accounts are significantly stickier
3. Contract: Two Year — strongest protection against churn
4. Internet Service: Fiber Optic — highest churn risk segment
5. Contract: One Year — moderate protection vs month-to-month

## Business Recommendations
1. **Prioritize contract upgrade campaigns** for month-to-month customers in their first year
2. **Invest in customer service quality** — top churn reason is support experience
3. **Address fiber optic product gaps** — speed and data offerings vs competitors
4. **Develop family plan promotions** — dependents is a strong retention signal

## Project Structure
```
churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Encoding, scaling, train/test split
│   └── 03_modeling.ipynb        # Model training, evaluation, SHAP
├── app/
│   └── app.py                   # Streamlit web application
├── models/
│   ├── best_model.pkl           # Saved Gradient Boosting model
│   ├── scaler.pkl               # Fitted StandardScaler
│   └── feature_names.pkl        # Feature names for inference
├── reports/figures/             # EDA and model visualizations
├── insights.md                  # Business insights documentation
├── requirements.txt
└── README.md
```

## Tech Stack
Python, pandas, scikit-learn, SHAP, Streamlit

## How to Run
```bash
# Clone the repo
git clone git@github.com:yourusername/churn-prediction.git
cd churn-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app/app.py
```

## Live Demo
*Coming soon — Streamlit Cloud deployment*

## Future Improvements
- Hyperparameter tuning to improve recall beyond 53%
- SMOTE oversampling to handle class imbalance
- Lower churn risk threshold from 70% to 60% for earlier intervention
- Connect to CRM for automated daily customer scoring
- Add usage trend features (charge increase over time)

## Author
Daffa Abiyyu
https://github.com/dabiyyu | www.linkedin.com/in/daffa-abiyyu-7a0125202
