# Business Insights — Churn Analysis

## Overview
- Overall churn rate: 27% (1,869 customers lost out of 7,043)

## Contract Type (Highest Leverage Finding)
- Month-to-month: 42.7% churn rate
- One year: 11.3% churn rate
- Two year: 2.8% churn rate
- → Month-to-month customers churn 15x more than two-year customers
- → Contract upgrade campaigns are likely the highest-ROI retention lever

## Tenure (When Customers Leave)
- 0-12 months is the highest churn period
- → First year is the critical retention window
- → Onboarding experience and early engagement programs matter most

## Monthly Charges (Revenue at Risk)
- Retained customers avg: $61.27/month
- Churned customers avg: $74.44/month
- Difference: $13.17/month more for churned customers
- → Higher-paying customers are leaving at a disproportionate rate
- → Revenue loss is worse than churn rate alone suggests

## Why Customers Leave (Churn Reasons)
1. Attitude of support person — service quality issue
2. Competitor offered higher download speeds — product gap
3. Competitor offered more data — product gap
- → Two distinct problem areas: customer service quality + product competitiveness
- → Short-term fix: customer service training
- → Long-term fix: product roadmap response to competitor offerings

## Data Quality Issues Found
- Total Charges: comma decimal separator — fixed
- Monthly Charges: comma decimal separator — fixed

## Internet Service
- Fiber optic: 41.9% churn rate
- DSL: 19.0% churn rate
- No internet: 7.4% churn rate
- → Fiber optic customers churn at 2x the rate of DSL despite paying more
- → Likely tied to competitor speed/data offerings (matches churn reasons #2 and #3)
- → Fiber optic retention should be a priority segment

## Customer Lifetime Value
- Retained customers avg CLTV: $4,491
- Churned customers avg CLTV: $4,149
- Total CLTV lost to churn: $7,755,256
- → Nearly $7.8M in lifetime value walked out the door
- → Even reducing churn by 5% would recover ~$387,000 in CLTV

## Model — Top 5 Predictive Features (SHAP)
1. Tenure Months — longer tenure strongly predicts retention
   → Confirms first year is the critical churn window
2. Dependents — customers with dependents churn less
   → Family accounts are stickier, worth targeting with family plans
3. Contract: Two Year — strongest contract protection against churn
   → Directly confirms the contract upgrade recommendation
4. Internet Service: Fiber Optic — fiber users at significantly higher churn risk
   → Confirms product gap vs competitors (speed/data offerings)
5. Contract: One Year — moderate protection vs month-to-month
   → Even moving customers from monthly to annual reduces risk significantly

## Model Performance
- Best model: Gradient Boosting
- ROC-AUC: 0.8501
- Churners correctly identified: 198/374 (53% recall)
- Future improvement: hyperparameter tuning to improve recall