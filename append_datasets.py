import json
import os

with open(r'd:\PySliceKit\examples\example.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Adult Dataset Markdown and Code
adult_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Classification Task: Adult Income\n",
        "Testing on the Adult Income dataset with real demographic slice columns (`sex` and `race`)."
    ]
}

adult_code1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.datasets import fetch_openml\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# Load Adult dataset (version 2 is standard)\n",
        "adult = fetch_openml(\"adult\", version=2, as_frame=True)\n",
        "df_adult = adult.frame.dropna()\n",
        "\n",
        "# Target: Predict if income >50K\n",
        "y_adult = (df_adult['class'] == '>50K').astype(int)\n",
        "X_adult = df_adult.drop(columns=['class'])\n",
        "\n",
        "# Fast preprocessing for RandomForest\n",
        "X_encoded = pd.get_dummies(X_adult)\n",
        "\n",
        "X_train_a, X_test_a, y_train_a, y_test_a, df_train_a, df_test_a = train_test_split(\n",
        "    X_encoded, y_adult, df_adult, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "clf_a = RandomForestClassifier(n_estimators=50, random_state=42)\n",
        "clf_a.fit(X_train_a, y_train_a)\n",
        "y_pred_a = clf_a.predict(X_test_a)"
    ]
}

adult_code2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "results_adult = pyslicekit.evaluate(\n",
        "    model=clf_a,\n",
        "    df=df_test_a,\n",
        "    y_true=y_test_a.values,\n",
        "    y_pred=y_pred_a,\n",
        "    slice_cols=['sex', 'race'],\n",
        "    metric='accuracy',\n",
        "    min_samples=30,\n",
        "    depth=2,\n",
        "    render_visuals=False # Turning off base visuals to focus on the correction chart\n",
        ")\n",
        "\n",
        "apply_correction(results_adult, method=\"fdr_bh\")\n",
        "fig_adult = render_correction_comparison(results_adult, method=\"fdr_bh\")\n",
        "fig_adult"
    ]
}

# Diabetes Dataset Markdown and Code
diabetes_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Regression Task: Diabetes Dataset\n",
        "Testing on the Diabetes dataset with generated categorical bins."
    ]
}

diabetes_code1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.datasets import load_diabetes\n",
        "from sklearn.linear_model import Ridge\n",
        "\n",
        "diabetes = load_diabetes(as_frame=True)\n",
        "df_d = diabetes.frame\n",
        "X_d = df_d.drop(columns=['target'])\n",
        "y_d = df_d['target']\n",
        "\n",
        "X_train_d, X_test_d, y_train_d, y_test_d, df_train_d, df_test_d = train_test_split(\n",
        "    X_d, y_d, df_d, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "# Discretize numeric columns into bins so PySliceKit can slice them\n",
        "df_test_d = df_test_d.copy()\n",
        "df_test_d['age_group'] = pd.qcut(df_test_d['age'], q=3, labels=['Young', 'Middle', 'Old'])\n",
        "df_test_d['bmi_group'] = pd.qcut(df_test_d['bmi'], q=3, labels=['Low', 'Med', 'High'])\n",
        "\n",
        "model_d = Ridge().fit(X_train_d, y_train_d)\n",
        "y_pred_d = model_d.predict(X_test_d)"
    ]
}

diabetes_code2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "results_diabetes = pyslicekit.evaluate(\n",
        "    model=model_d,\n",
        "    df=df_test_d,\n",
        "    y_true=y_test_d.values,\n",
        "    y_pred=y_pred_d,\n",
        "    slice_cols=['age_group', 'bmi_group'],\n",
        "    metric='mae',\n",
        "    min_samples=15,\n",
        "    depth=2,\n",
        "    render_visuals=False\n",
        ")\n",
        "\n",
        "apply_correction(results_diabetes, method=\"fdr_bh\")\n",
        "fig_diabetes = render_correction_comparison(results_diabetes, method=\"fdr_bh\")\n",
        "fig_diabetes"
    ]
}

nb['cells'].extend([
    adult_md, adult_code1, adult_code2,
    diabetes_md, diabetes_code1, diabetes_code2
])

with open(r'd:\PySliceKit\examples\example.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
