import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml, load_diabetes
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
import pyslicekit
from pyslicekit.correction import apply_correction

def print_summary(name, results):
    apply_correction(results, method="fdr_bh")
    fdr_survived = sum(1 for r in results if r.extra.get('is_significant_corrected', False))
    
    for r in results:
        r.extra.pop('is_significant_corrected', None)
        r.extra.pop('p_value_corrected', None)
        r.extra.pop('correction_method', None)
    apply_correction(results, method="bonferroni")
    bonf_survived = sum(1 for r in results if r.extra.get('is_significant_corrected', False))

    eligible = [r for r in results if r.p_value is not None]
    sig_before = sum(1 for r in eligible if r.is_significant)
    print(f"{name}: Eligible={len(eligible)}, Sig Before={sig_before}, Sig After FDR={fdr_survived}, Sig After Bonf={bonf_survived}")
    if name == "California Housing":
        for r in eligible:
            if r.is_significant:
                print(f"  [Cal] Label={r.label}, p={r.p_value:.5f}, Bonf_survived={r.extra.get('is_significant_corrected', False)}")

# California Housing
cali = fetch_california_housing(as_frame=True)
df = cali.frame
np.random.seed(42)
df['ocean_proximity'] = np.random.choice(['INLAND', 'NEAR BAY', 'NEAR OCEAN', '<1H OCEAN'], size=len(df))
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train.drop(columns=['ocean_proximity']), y_train)
y_pred = model.predict(X_test.drop(columns=['ocean_proximity']))
results_reg = pyslicekit.evaluate(model, X_test, y_test.values, y_pred, slice_cols=['HouseAge', 'ocean_proximity'], metric='mae', min_samples=50, depth=2, render_visuals=False)
print_summary("California Housing", results_reg)

# Breast Cancer
cancer = load_breast_cancer(as_frame=True)
df_c = cancer.frame
X_c = df_c.drop(columns=['target'])
y_c = df_c['target']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.25, random_state=42)
clf = LogisticRegression(max_iter=5000).fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
results_clf = pyslicekit.evaluate(clf, X_test_c, y_test_c.values, y_pred_c, slice_cols=['mean radius', 'mean texture'], metric='f1', min_samples=10, depth=2, render_visuals=False)
print_summary("Breast Cancer", results_clf)

# Adult Income
adult = fetch_openml(data_id=1590, as_frame=True)
df_a = adult.frame.dropna()
X_a = df_a.drop(columns=['class'])
y_a = (df_a['class'] == '>50K').astype(int)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size=0.2, random_state=42)
numeric_cols = X_train_a.select_dtypes(include=[np.number]).columns
clf_a = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_a[numeric_cols], y_train_a)
y_pred_a = clf_a.predict(X_test_a[numeric_cols])
results_adult = pyslicekit.evaluate(clf_a, X_test_a, y_test_a.values, y_pred_a, slice_cols=['sex', 'race'], metric='accuracy', min_samples=30, depth=2, render_visuals=False)
print_summary("Adult Income", results_adult)

# Diabetes
diabetes = load_diabetes(as_frame=True)
df_d = diabetes.frame
X_d = df_d.drop(columns=['target'])
y_d = df_d['target']
X_d['age_group'] = pd.qcut(X_d['age'], q=3, labels=['Young', 'Middle', 'Old'])
X_d['bmi_group'] = pd.qcut(X_d['bmi'], q=3, labels=['Low', 'Med', 'High'])
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
model_d = Ridge().fit(X_train_d.drop(columns=['age_group', 'bmi_group']), y_train_d)
y_pred_d = model_d.predict(X_test_d.drop(columns=['age_group', 'bmi_group']))
results_diabetes = pyslicekit.evaluate(model_d, X_test_d, y_test_d.values, y_pred_d, slice_cols=['age_group', 'bmi_group'], metric='mae', min_samples=15, depth=2, render_visuals=False)
print_summary("Diabetes", results_diabetes)
