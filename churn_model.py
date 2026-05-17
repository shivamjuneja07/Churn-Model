"""
Customer Churn Prediction
=========================
Telecom dataset with 7,043 customers.
Goal: predict which customers are likely to churn so the business
can take action before they leave.

Pipeline:
  1. Load and inspect data
  2. Exploratory data analysis
  3. Feature engineering
  4. Train/test split
  5. Neural network model (Keras)
  6. Evaluation and business insights
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PALETTE = {"Yes": "#e74c3c", "No": "#2ecc71"}
plt.style.use("seaborn-v0_8-whitegrid")

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/model", exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("Churn.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows   : {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)
churn_counts = df["Churn"].value_counts()
churn_rate = (df["Churn"] == "Yes").mean()
print(churn_counts)
print(f"\nOverall churn rate: {churn_rate:.1%}")

# -- Figure 1: churn overview -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Customer Churn Overview", fontsize=15, fontweight="bold", y=1.02)

# churn pie
axes[0].pie(
    churn_counts,
    labels=["Retained", "Churned"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 12},
)
axes[0].set_title("Churn Split")

# contract type
contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).reset_index()
contract_churn.columns = ["Contract", "Churn Rate (%)"]
axes[1].bar(
    contract_churn["Contract"],
    contract_churn["Churn Rate (%)"],
    color=["#e74c3c", "#f39c12", "#2ecc71"],
    edgecolor="white",
)
axes[1].set_title("Churn Rate by Contract Type")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_ylim(0, 55)
for bar, val in zip(axes[1].patches, contract_churn["Churn Rate (%)"]):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{val:.1f}%",
        ha="center",
        fontsize=10,
    )

# payment method
pay_churn = df.groupby("Payment Method")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).reset_index()
pay_churn.columns = ["Payment Method", "Churn Rate (%)"]
pay_churn = pay_churn.sort_values("Churn Rate (%)", ascending=True)
axes[2].barh(
    pay_churn["Payment Method"],
    pay_churn["Churn Rate (%)"],
    color="#3498db",
    edgecolor="white",
)
axes[2].set_title("Churn Rate by Payment Method")
axes[2].set_xlabel("Churn Rate (%)")

plt.tight_layout()
plt.savefig("outputs/figures/01_churn_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/01_churn_overview.png")


# -- Figure 2: numerical distributions ---------------------------------------
numerical_cols = ["tenure", "Monthly Charges", "Total Charges"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Numerical Features by Churn Status", fontsize=15, fontweight="bold", y=1.02
)

for ax, col in zip(axes, numerical_cols):
    for label, color in PALETTE.items():
        subset = df[df["Churn"] == label][col].dropna()
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(title="Churned")

plt.tight_layout()
plt.savefig(
    "outputs/figures/02_numerical_distributions.png", dpi=150, bbox_inches="tight"
)
plt.close()
print("Saved: outputs/figures/02_numerical_distributions.png")


# -- Figure 3: segment churn rates -------------------------------------------
segments = {
    "Internet Service": df.groupby("Internet Service")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ),
    "Senior Citizen": df.groupby("Senior Citizen")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ),
    "Partner": df.groupby("Partner")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ),
    "Dependents": df.groupby("Dependents")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Churn Rate by Customer Segment", fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax, (title, data) in zip(axes, segments.items()):
    bars = ax.bar(
        data.index.astype(str),
        data.values,
        color="#3498db",
        edgecolor="white",
    )
    ax.set_title(title)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, data.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig("outputs/figures/03_segment_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/03_segment_churn.png")


# -- Figure 4: correlation heatmap -------------------------------------------
df_encoded = pd.get_dummies(df.drop("Customer ID", axis=1))
df_encoded["Churn"] = (df["Churn"] == "Yes").astype(int)

corr_with_churn = (
    df_encoded.corr()["Churn"]
    .drop("Churn")
    .sort_values()
)

top_pos = corr_with_churn.tail(8)
top_neg = corr_with_churn.head(8)
top_features = pd.concat([top_neg, top_pos])

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in top_features.values]
bars = ax.barh(top_features.index, top_features.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(
    "Top Feature Correlations with Churn", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("outputs/figures/04_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/04_feature_correlations.png")


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df_model = df.copy()

# tenure buckets
df_model["tenure_group"] = pd.cut(
    df_model["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
)

# charge ratio: monthly spend relative to total relationship value
df_model["charge_ratio"] = df_model["Monthly Charges"] / (
    df_model["Total Charges"] + 1
)

# number of add-on services
service_cols = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]
df_model["service_count"] = sum(
    (df_model[col] == "Yes").astype(int) for col in service_cols
)

print(f"New features added: tenure_group, charge_ratio, service_count")
print(f"\nService count distribution:\n{df_model['service_count'].value_counts().sort_index()}")

# one-hot encode all categoricals
X = pd.get_dummies(df_model.drop(["Churn", "Customer ID"], axis=1))
y = (df["Churn"] == "Yes").astype(int)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")


# =============================================================================
# 4. TRAIN / TEST SPLIT + SCALING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")


# =============================================================================
# 5. NEURAL NETWORK MODEL
# =============================================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# class imbalance: 73% retained, 27% churned
class_weight = {
    0: 1.0,
    1: (y_train == 0).sum() / (y_train == 1).sum(),
}
print(f"Class weight for churn class: {class_weight[1]:.2f}")

model = Sequential(
    [
        Dense(128, activation="relu", input_dim=X_train_scaled.shape[1]),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_auc", patience=20, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
]

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)


# =============================================================================
# 6. EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {roc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Retained','Churned'])}")

# -- Figure 5: training history -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

axes[0].plot(history.history["loss"], label="Train Loss", color="#3498db")
axes[0].plot(history.history["val_loss"], label="Val Loss", color="#e74c3c")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["auc"], label="Train AUC", color="#3498db")
axes[1].plot(history.history["val_auc"], label="Val AUC", color="#e74c3c")
axes[1].set_title("AUC")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/figures/05_training_history.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/05_training_history.png")

# -- Figure 6: confusion matrix + ROC ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance", fontsize=14, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Retained", "Churned"],
    yticklabels=["Retained", "Churned"],
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC Curve (AUC = {roc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/figures/06_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/06_model_performance.png")

# -- Figure 7: churn probability distribution ---------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    y_prob[y_test == 0],
    bins=40,
    alpha=0.6,
    color="#2ecc71",
    label="Retained",
    density=True,
)
ax.hist(
    y_prob[y_test == 1],
    bins=40,
    alpha=0.6,
    color="#e74c3c",
    label="Churned",
    density=True,
)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision threshold (0.5)")
ax.set_title("Predicted Churn Probability by Actual Label", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Probability of Churn")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/07_probability_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/figures/07_probability_distribution.png")


# =============================================================================
# 7. BUSINESS INSIGHT: HIGH-RISK CUSTOMERS
# =============================================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHT: HIGH-RISK CUSTOMERS")
print("=" * 60)

df_test = X_test.copy()
df_test["churn_probability"] = y_prob
df_test["actual_churn"] = y_test.values
df_test["risk_tier"] = pd.cut(
    df_test["churn_probability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

tier_summary = (
    df_test.groupby("risk_tier", observed=True)
    .agg(customer_count=("churn_probability", "count"),
         avg_churn_prob=("churn_probability", "mean"),
         actual_churn_rate=("actual_churn", "mean"))
    .reset_index()
)
print(tier_summary.to_string(index=False))

high_risk_count = (df_test["risk_tier"] == "High Risk").sum()
print(
    f"\n{high_risk_count} customers ({high_risk_count/len(df_test):.1%} of test set) "
    f"are flagged as high risk."
)

# =============================================================================
# 8. SAVE MODEL
# =============================================================================

model.save("outputs/model/churn_model.keras")
print("\nModel saved to outputs/model/churn_model.keras")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
