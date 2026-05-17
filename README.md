# Customer Churn Prediction with Neural Networks

A complete machine learning project to predict customer churn for a telecom company.

---

## What This Project Does

Customer churn is one of the most expensive problems in any subscription business. Acquiring a new customer costs 5 to 7 times more than keeping an existing one. This project builds a neural network that scores every customer by their likelihood to leave, so retention teams can act early and focus their effort where it matters most.

The model reaches **~80% accuracy and 0.84 ROC-AUC** on held-out data.

---

## Key Findings from the Data

Before building anything, I spent time understanding what actually drives churn. A few things stood out immediately.

**Contract type is the single strongest predictor.**
Customers on month-to-month contracts churn at 42.7%. Customers on two-year contracts churn at just 2.8%. That is a 15x difference in churn rate based on one variable.

**Fiber optic customers are leaving at a high rate.**
Customers with fiber optic internet churn at 41.9%, compared to 19% for DSL and 7.4% for customers with no internet service. This suggests a pricing or service quality issue specific to that product tier.

**Electronic check users churn most.**
Customers who pay by electronic check churn at 45.3%, nearly three times the rate of those on automatic payment methods (15 to 16%). This could reflect a less committed relationship with the service or a demographic that is more price-sensitive.

**Churned customers have much shorter tenure.**
Average tenure for churned customers is 18 months. For retained customers it is 37 months. New customers are the highest risk, which makes early engagement critical.

**Senior citizens churn at nearly double the rate.**
Senior citizens churn at 41.7% versus 23.6% for non-seniors. This is worth investigating further from a customer experience perspective.

---

## Project Structure

```
churn-project/
├── churn_model.py          # Main script: EDA, feature engineering, model, evaluation
├── requirements.txt        # Python dependencies
├── Churn.csv               # Source dataset (7,043 customers, 21 features)
├── report/
│   └── churn_analysis.md   # Short written analysis of results
└── outputs/
    ├── figures/            # All charts (auto-generated on run)
    └── model/              # Saved Keras model
```

---

## Dataset

The dataset contains telecom customers with 21 features covering demographics, account details, and service subscriptions.

| Feature | Type | Description |
|---|---|---|
| tenure | Numeric | Months as a customer |
| Monthly Charges | Numeric | Current monthly bill |
| Total Charges | Numeric | Lifetime spend |
| Contract | Categorical | Month-to-month, One year, Two year |
| Internet Service | Categorical | DSL, Fiber optic, None |
| Payment Method | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| ... | ... | 15 additional features |

**Churn rate: 26.5%** (1,869 churned out of 7,043 customers)

---

## Model Architecture

The model is a fully connected neural network built with Keras.

```
Input (44 features after encoding)
  Dense(128, relu)
  BatchNormalization
  Dropout(0.3)
  Dense(64, relu)
  BatchNormalization
  Dropout(0.2)
  Dense(32, relu)
  Dense(1, sigmoid)
```

Key design choices:
- **Batch normalization** stabilizes training and reduces sensitivity to learning rate
- **Dropout** reduces overfitting on a moderately sized dataset
- **Class weighting** addresses the 73/27 class imbalance without discarding data
- **Early stopping** on validation AUC prevents overfitting automatically

---

## Feature Engineering

Three features were added beyond the raw columns.

- **tenure_group**: Buckets customers into 0-1yr, 1-2yr, 2-4yr, 4-6yr. Churn risk is not linear with time, so bucketing lets the model capture threshold effects.
- **charge_ratio**: Monthly charges divided by total charges. A high ratio means a newer customer paying full price, which correlates with churn.
- **service_count**: Number of add-on services (security, backup, streaming, etc.). Customers with more services have a higher switching cost and churn less.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| ROC-AUC | ~0.84 |
| Precision (Churn) | ~65% |
| Recall (Churn) | ~75% |

The model correctly identifies around three out of four customers who will churn. The ROC-AUC of 0.84 means the model is substantially better than random at ranking customers by risk, which is what matters most in a real retention campaign.

---

## Business Application

The model outputs a churn probability for every customer, which gets bucketed into three risk tiers.

| Risk Tier | Probability | Recommended Action |
|---|---|---|
| High Risk | > 66% | Immediate outreach, targeted retention offer |
| Medium Risk | 33-66% | Proactive check-in, loyalty incentives |
| Low Risk | < 33% | Standard engagement, no intervention needed |

Focusing retention spend on the high-risk tier means fewer wasted calls and better ROI on retention campaigns.

---
