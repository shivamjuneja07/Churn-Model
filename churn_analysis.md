# Churn Analysis Report

**Dataset:** Telecom customer base, 7,043 records, 21 features  
**Target:** Binary churn label (Yes / No)  
**Overall churn rate:** 26.5%

---

## What the Data Shows

### Contract Type Drives Everything

The strongest single predictor of churn is how the customer signed up.

| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

A customer on a month-to-month contract is 15 times more likely to churn than one on a two-year contract. The business implication is clear: getting customers to commit to longer contracts dramatically reduces churn risk. This could be worth a significant discount at signup.

### Fiber Optic is a Problem

Customers on fiber optic internet churn at 41.9%, more than twice the rate of DSL customers (19.0%). This is a product-specific signal. Either the price point is too high for the value delivered, or there are service quality issues not captured in this dataset. Either way, fiber optic customers should be a priority segment for retention.

### Payment Method Signals Commitment

| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

Customers on automatic payment methods churn at roughly one-third the rate of those paying by electronic check. One interpretation is that automatic payment customers are more committed. Another is that the act of switching to autopay itself selects for customers who intend to stay. Nudging customers toward autopay could have a meaningful effect.

### New Customers Are the Highest Risk

Churned customers have an average tenure of 18 months. Retained customers average 37 months. The first year of a customer relationship is the most critical period. Onboarding programs, early check-ins, and introductory loyalty rewards all have a strong justification here.

### Senior Citizens Need Attention

Senior citizens (age 65+) churn at 41.7% versus 23.6% for other customers. This is a nearly 2x difference. It is worth investigating whether the product experience, pricing, or support quality differs meaningfully for this group.

### Add-On Services Are a Retention Tool

Customers with more add-on services (security, backup, tech support, streaming) churn at lower rates. Each additional service raises the cost of switching. Upselling is not just a revenue play; it is also a retention strategy.

---

## Model Performance

The neural network achieves a ROC-AUC of approximately 0.84 on held-out test data. This means that if you pick a random churned customer and a random retained customer, the model ranks the churned customer as higher risk 84% of the time.

Recall on the churn class is around 75%, meaning the model catches three out of four customers who will leave. This is the metric that matters most in practice: missing a churner is more costly than a false alarm.

The confusion between the classes is expected given the imbalance. The model is calibrated with class weighting so that churn recall does not collapse to near zero, which is a common failure mode on imbalanced datasets.

---

## Recommendations

1. **Prioritize long-term contract conversion at acquisition.** The churn rate difference between month-to-month and two-year contracts is the largest signal in the dataset. Even a small discount to lock in a longer contract pays for itself quickly.

2. **Investigate fiber optic pricing and quality.** The 42% churn rate in that segment is high enough to suggest a product problem, not just a customer mix problem.

3. **Incentivize autopay enrollment.** The correlation between automatic payments and retention is strong. A small bill credit for autopay enrollment is worth testing.

4. **Build a 90-day early warning program.** Given that new customers churn at the highest rates, a structured outreach program in the first three months could move the needle significantly.

5. **Deploy the model in a risk-tiered retention workflow.** Use the model output to allocate retention resources: proactive outreach for high-risk customers, lighter-touch engagement for medium-risk, and standard service for low-risk.

---

## Limitations

- The dataset is cross-sectional (a snapshot in time). A time-series approach that uses behavioral changes over months would likely perform better in production.
- No cost data is included. Optimizing for recall vs. precision should ultimately be driven by the revenue at risk per customer and the cost of a retention call.
- The model has not been evaluated for demographic fairness. The senior citizen finding warrants a closer look to ensure interventions are equitable.
