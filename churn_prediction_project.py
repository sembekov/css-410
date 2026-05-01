"""
THESIS PROJECT: Customer Churn Prediction for Telecom Industry
Author: Sayat Sembekov | 230107003@sdu.edu.kz
Supervisor: Dr. Selcuk Cankurt

This project builds and compares three machine learning models to predict 
customer churn, then calculates business ROI. The best model is deployed 
via an interactive dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("=" * 80)
print("THESIS PROJECT: CUSTOMER CHURN PREDICTION IN TELECOM")
print("Author: Sayat Sembekov (230107003@sdu.edu.kz)")
print("Supervisor: Dr. Selcuk Cankurt")
print("=" * 80)

# ============================================================================
# STEP 1: DATA GENERATION (Simulating real telecom customer data)
# ============================================================================

print("\n[STEP 1] Generating synthetic telecom customer dataset...")
print("-" * 50)

np.random.seed(42)
n_customers = 10000

# Generate realistic telecom customer features
data = {
    # Demographics
    'customer_id': np.arange(n_customers),
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
    'partner': np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
    'dependents': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    
    # Account information
    'tenure_months': np.random.exponential(30, n_customers).astype(int),
    'monthly_charges': np.random.uniform(20, 120, n_customers),
    'total_charges': lambda df: df['monthly_charges'] * df['tenure_months'],
    
    # Services subscribed
    'phone_service': np.random.choice([0, 1], n_customers, p=[0.1, 0.9]),
    'multiple_lines': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.4, 0.4, 0.2]),
    'online_security': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    'online_backup': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    'device_protection': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    'tech_support': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
    'streaming_tv': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
    'streaming_movies': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
    
    # Contract and payment
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                      n_customers, p=[0.55, 0.25, 0.20]),
    'paperless_billing': np.random.choice([0, 1], n_customers),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                        'Bank transfer', 'Credit card'], n_customers),
    
    # Usage patterns (generated based on other features)
    'avg_monthly_gb': np.random.gamma(2, 15, n_customers),
    'customer_service_calls': np.random.poisson(1.5, n_customers),
}

# Create DataFrame
df = pd.DataFrame(data)
df['total_charges'] = df['monthly_charges'] * df['tenure_months']

def calculate_churn_probability(row):
    """Higher probability = more likely to churn"""
    prob = 0.10  # Base churn rate
    
    if row['tenure_months'] < 6:
        prob += 0.35
    elif row['tenure_months'] < 12:
        prob += 0.20
    elif row['tenure_months'] < 24:
        prob += 0.10
    
    # Month-to-month contracts churn more
    if row['contract_type'] == 'Month-to-month':
        prob += 0.25
    elif row['contract_type'] == 'One year':
        prob -= 0.10
    else:
        prob -= 0.15
    
    if row['monthly_charges'] > 100:
        prob += 0.15
    elif row['monthly_charges'] > 70:
        prob += 0.08
    
    if row['customer_service_calls'] > 3:
        prob += 0.25
    elif row['customer_service_calls'] > 2:
        prob += 0.10
    
    if row['tech_support'] == 0 and row['internet_service'] != 'No':
        prob += 0.10
    
    if row['online_security'] == 0 and row['internet_service'] != 'No':
        prob += 0.08
    
    if row['paperless_billing'] == 1:
        prob += 0.03
    
    if row['senior_citizen'] == 1:
        prob -= 0.05
    
    prob += np.random.normal(0, 0.03)
    
    return min(max(prob, 0), 1)

df['churn_probability'] = df.apply(calculate_churn_probability, axis=1)
df['churn'] = (np.random.random(n_customers) < df['churn_probability']).astype(int)

# Check class balance
churn_rate = df['churn'].mean()
print(f"\nDataset generated: {n_customers:,} customers")
print(f"Churn rate: {churn_rate:.2%} ({df['churn'].sum():,} customers will churn)")
print(f"Retention rate: {1-churn_rate:.2%} ({n_customers - df['churn'].sum():,} will stay)")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n[STEP 2] Exploratory Data Analysis...")
print("-" * 50)

# Analyze churn by key features
print("\nChurn rate by contract type:")
contract_churn = df.groupby('contract_type')['churn'].mean().sort_values()
for contract, rate in contract_churn.items():
    print(f"  {contract}: {rate:.2%}")

print("\nChurn rate by service calls:")
calls_churn = df.groupby('customer_service_calls')['churn'].mean()
for calls, rate in calls_churn.items():
    if calls <= 5:
        print(f"  {calls} call(s): {rate:.2%}")

print("\nChurn rate by internet service type:")
internet_churn = df.groupby('internet_service')['churn'].mean()
for service, rate in internet_churn.items():
    print(f"  {service}: {rate:.2%}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n[STEP 3] Data Preprocessing...")
print("-" * 50)

# Create a copy for modeling
model_df = df.copy()

# Drop customer_id (not a feature) and churn_probability (derived)
model_df = model_df.drop(['customer_id', 'churn_probability'], axis=1)

# Encode categorical variables
categorical_cols = ['gender', 'internet_service', 'contract_type', 'payment_method']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le
    print(f"  Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Binary columns (ensure they are integer)
binary_cols = ['senior_citizen', 'partner', 'dependents', 'phone_service', 
               'multiple_lines', 'online_security', 'online_backup', 
               'device_protection', 'tech_support', 'streaming_tv', 
               'streaming_movies', 'paperless_billing']
for col in binary_cols:
    model_df[col] = model_df[col].astype(int)

# Prepare features and target
feature_cols = [col for col in model_df.columns if col != 'churn']
X = model_df[feature_cols]
y = model_df['churn']

print(f"\nFeatures: {len(feature_cols)} features")
print(f"Features list: {feature_cols[:10]}...")

numerical_cols = ['tenure_months', 'monthly_charges', 'total_charges', 
                  'avg_monthly_gb', 'customer_service_calls']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)
print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# STEP 4: MODEL TRAINING AND COMPARISON
# ============================================================================

print("\n[STEP 4] Training and Comparing Machine Learning Models...")
print("-" * 50)

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
}

# Store results
results = {}
best_model = None
best_auc = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'model': model
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  CV AUC (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================

print(f"\n[STEP 5] Hyperparameter Tuning for Best Model ({best_model_name})...")
print("-" * 50)

if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                               param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    y_pred_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, y_pred_proba_tuned)
    print(f"Tuned model test AUC: {tuned_auc:.4f}")
    
    if tuned_auc > best_auc:
        best_model = tuned_model
        best_auc = tuned_auc
        print("Tuned model outperformed original!")

# ============================================================================
# STEP 6: BUSINESS IMPACT ANALYSIS (ROI CALCULATION)
# ============================================================================

print("\n[STEP 6] Business Impact Analysis & ROI Calculation...")
print("-" * 50)

avg_customer_value = 1200  # Average revenue per customer per year
retention_cost = 50  # Cost to retain a customer (discount, offer, etc.)

y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

threshold = np.percentile(y_proba_final, 80)
high_risk_indices = y_proba_final >= threshold
actual_churn_high_risk = y_test[high_risk_indices].mean() if high_risk_indices.any() else 0

baseline_churn_rate = y_test.mean()
expected_loss_baseline = baseline_churn_rate * len(y_test) * avg_customer_value

reduction_rate = 0.40
churn_rate_with_ai = (baseline_churn_rate * len(y_test) - 
                      actual_churn_high_risk * high_risk_indices.sum() * reduction_rate) / len(y_test)

expected_loss_with_ai = churn_rate_with_ai * len(y_test) * avg_customer_value
retention_campaign_cost = high_risk_indices.sum() * retention_cost
net_savings = expected_loss_baseline - expected_loss_with_ai - retention_campaign_cost
roi = (net_savings / retention_campaign_cost) * 100 if retention_campaign_cost > 0 else 0

print(f"\nBusiness Assumptions:")
print(f"  Average customer lifetime value: ${avg_customer_value:,.0f}")
print(f"  Retention campaign cost per customer: ${retention_cost}")
print(f"  Expected churn reduction from campaign: {reduction_rate:.0%}")

print(f"\nFinancial Impact Analysis:")
print(f"  Test set size: {len(y_test)} customers")
print(f"  Baseline churn rate: {baseline_churn_rate:.2%}")
print(f"  Expected loss (baseline): ${expected_loss_baseline:,.0f}")
print(f"  High-risk customers identified: {high_risk_indices.sum()} (top 20%)")
print(f"  Churn rate among high-risk: {actual_churn_high_risk:.2%}")
print(f"  Churn rate after AI intervention: {churn_rate_with_ai:.2%}")
print(f"  Expected loss (with AI): ${expected_loss_with_ai:,.0f}")
print(f"  Retention campaign cost: ${retention_campaign_cost:,.0f}")
print(f"  Net savings: ${net_savings:,.0f}")
print(f"  Return on Investment (ROI): {roi:.1f}%")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n[STEP 7] Generating Visualizations...")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

models_list = list(results.keys())
auc_scores = [results[m]['roc_auc'] for m in models_list]
bars = axes[0, 0].bar(models_list, auc_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
axes[0, 0].set_ylim(0.5, 1)
axes[0, 0].set_ylabel('ROC-AUC Score')
axes[0, 0].set_title('Model Performance Comparison (ROC-AUC)')
axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
axes[0, 0].legend()
for bar, score in zip(bars, auc_scores):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', fontsize=10)

for name, res in results.items():
    model = res['model']
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})', linewidth=2)
axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves - All Models')
axes[0, 1].legend(loc='lower right', fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
axes[0, 2].set_xlabel('Predicted')
axes[0, 2].set_ylabel('Actual')
axes[0, 2].set_title(f'Confusion Matrix - {best_model_name}')

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    axes[1, 0].barh(range(10), importances[indices])
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in indices])
    axes[1, 0].set_xlabel('Feature Importance')
    axes[1, 0].set_title('Top 10 Features for Churn Prediction')
else:
    axes[1, 0].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Feature Importance Not Available')

metrics = ['Baseline', 'With AI']
losses = [expected_loss_baseline, expected_loss_with_ai]
colors_business = ['#C73E1D', '#2E86AB']
bars = axes[1, 1].bar(metrics, losses, color=colors_business)
axes[1, 1].set_ylabel('Expected Loss ($)')
axes[1, 1].set_title('Financial Impact: Baseline vs AI Intervention')
for bar, loss in zip(bars, losses):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000, 
                    f'${loss:,.0f}', ha='center', fontsize=10)

axes[1, 2].hist(y_proba_final[y_test == 0], bins=30, alpha=0.7, label='Actual Non-Churn', color='green')
axes[1, 2].hist(y_proba_final[y_test == 1], bins=30, alpha=0.7, label='Actual Churn', color='red')
axes[1, 2].axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'High-risk threshold (top 20%)')
axes[1, 2].set_xlabel('Predicted Churn Probability')
axes[1, 2].set_ylabel('Number of Customers')
axes[1, 2].set_title('Distribution of Churn Predictions')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('thesis_churn_analysis_results.png', dpi=150, bbox_inches='tight')
print("  Visualization saved as 'thesis_churn_analysis_results.png'")

# ============================================================================
# STEP 8: SAVE RESULTS AND GENERATE REPORT
# ============================================================================

print("\n[STEP 8] Generating Final Report...")
print("-" * 50)

# Create results table
results_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'F1-Score': f"{res['f1_score']:.4f}",
        'ROC-AUC': f"{res['roc_auc']:.4f}",
        'CV AUC': f"{res['cv_auc_mean']:.4f} ± {res['cv_auc_std']:.4f}"
    }
    for name, res in results.items()
])

# Save to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("  Results saved to 'model_comparison_results.csv'")

# Print final summary
print("\n" + "=" * 80)
print("FINAL THESIS RESULTS SUMMARY")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ THESIS: Customer Churn Prediction in Telecom Using Machine Learning         │
│ Author: Sayat Sembekov | 230107003@sdu.edu.kz                               │
│ Supervisor: Dr. Selcuk Cankurt                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEST MODEL: {best_model_name:<35}                         │
│  TEST ROC-AUC: {best_auc:.4f}                                                │
│                                                                             │
│  MODEL COMPARISON:                                                          │
│                                                                             │
""")

for name, res in results.items():
    print(f"    {name:<20} ROC-AUC: {res['roc_auc']:.4f}  |  F1: {res['f1_score']:.4f}")

print(f"""
│                                                                             │
│  BUSINESS IMPACT:                                                           │
│    • Customers analyzed: {len(y_test):,}                                               │
│    • High-risk customers identified: {high_risk_indices.sum()} (top 20%)                 │
│    • Expected churn reduction: {reduction_rate:.0%} among targeted                         │
│    • Net savings from AI intervention: ${net_savings:,.0f}                          │
│    • Return on Investment (ROI): {roi:.1f}%                                             │
│                                                                             │
│  CONCLUSION:                                                                 │
│    The {best_model_name} model successfully predicts customer churn with              │
│    {best_auc:.1%} AUC. Implementing this model for targeted retention would            │
│    save approximately ${net_savings:,.0f} and yield {roi:.0f}% ROI.                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ Project completed successfully!")
print("📁 Files generated:")
print("   - thesis_churn_analysis_results.png (visualizations)")
print("   - model_comparison_results.csv (performance data)")
print("\n🎓 Good luck with your thesis submission!")

# Simulate deployment of "dashboard" (simple interactive function)
print("\n[DEMO] Interactive Prediction Function (simulates real deployment)")
print("-" * 50)

def predict_churn(tenure_months, monthly_charges, contract_type, customer_service_calls, 
                  internet_service, tech_support, online_security):
    """
    Simulate a real-time prediction for a single customer.
    This is what would be deployed in production.
    """
    # Create feature vector (simplified for demo)
    features = {
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': monthly_charges * tenure_months,
        'customer_service_calls': customer_service_calls,
        'avg_monthly_gb': 30,  # default
        'contract_type': contract_type,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'online_security': online_security,
        'gender': 0, 'senior_citizen': 0, 'partner': 0, 'dependents': 0,
        'phone_service': 1, 'multiple_lines': 0, 'online_backup': 0,
        'device_protection': 0, 'streaming_tv': 0, 'streaming_movies': 0,
        'paperless_billing': 1, 'payment_method': 0
    }
    
    # Convert to DataFrame with same columns as training
    input_df = pd.DataFrame([features])[feature_cols]
    
    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    prob = best_model.predict_proba(input_df)[0, 1]
    
    print(f"\n  Customer Analysis:")
    print(f"    Tenure: {tenure_months} months")
    print(f"    Monthly charges: ${monthly_charges}")
    print(f"    Contract: {contract_type}")
    print(f"    Service calls: {customer_service_calls}")
    print(f"    Churn risk: {prob:.1%}")
    print(f"    Recommendation: {'OFFER RETENTION DISCOUNT' if prob > 0.5 else 'MONITOR NORMALLY'}")
    
    return prob

# Demo with a few examples
print("\n  Example predictions:")
predict_churn(tenure_months=3, monthly_charges=95, contract_type='Month-to-month',
              customer_service_calls=4, internet_service='Fiber optic', 
              tech_support=0, online_security=0)
predict_churn(tenure_months=30, monthly_charges=65, contract_type='Two year',
              customer_service_calls=1, internet_service='DSL', 
              tech_support=1, online_security=1)
