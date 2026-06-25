import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_iris, load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import pyslicekit

def test_breast_cancer_binary_classification():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=model.predict(X_test),
        slice_cols=["mean radius"],
        metric="f1",
        render_visuals=False
    )
    assert len(results) > 0

def test_california_housing_regression():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.sample(1000, random_state=42) # subset for speed
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=model.predict(X_test),
        slice_cols=["HouseAge", "MedInc"],
        metric="mae",
        depth=2,
        render_visuals=False
    )
    assert len(results) > 0

def test_iris_multiclass_binarized():
    data = load_iris(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    # Binarize target to Class 0 vs Rest
    y = (df['target'] == 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=model.predict(X_test),
        slice_cols=["sepal length (cm)"],
        metric="accuracy",
        render_visuals=False
    )
    assert len(results) > 0

def test_diabetes_regression():
    data = load_diabetes(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=model.predict(X_test),
        slice_cols=["age", "sex"],
        metric="rmse",
        render_visuals=False
    )
    assert len(results) > 0

def test_wine_classification_binarized():
    data = load_wine(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = (df['target'] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=model.predict(X_test),
        slice_cols=["alcohol", "malic_acid"],
        metric="accuracy",
        depth=1,
        render_visuals=False
    )
    assert len(results) > 0
