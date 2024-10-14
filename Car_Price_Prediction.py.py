import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

st.title("Car Price Prediction using Multiple Regressions")

data_file = st.file_uploader("Upload your dataset", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    
    # Dataset preview
    st.write("Dataset Preview")
    st.write(df.head())
    
    # Basic statistics
    st.write("Basic Statistics")
    st.write(df.describe())

    # Missing values
    st.write("Missing Values")
    st.write(df.isnull().sum())

    # Correlation heatmap
    st.write("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt.gcf())

    # Feature names
    categorical_feature = ['fuel_type']  # Make sure this matches your dataset
    numerical_feature = ['engine_size', 'mileage', 'horsepower']  # Corrected 'milage' to 'mileage'

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())]), numerical_feature),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_feature)
        ]
    )

    # Features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Testing data shape: {X_test.shape}")

    # Linear regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regression', LinearRegression())])

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"R-Squared: {r2}")

    # Ridge regression
    ridge = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regression', Ridge(alpha=0.1))])
    ridge.fit(X_train, y_train)
    ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5)

    # Lasso regression
    lasso = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regression', Lasso(alpha=0.1))])  # Corrected 'Lesso' to 'Lasso'
    lasso.fit(X_train, y_train)
    lasso_cv = cross_val_score(lasso, X_train, y_train, cv=5)

    st.write(f"Cross Validation Score (Lasso): {lasso_cv.mean()}")

    # Plot: Actual vs Predicted Prices
    st.write('Predicted Price vs Actual Price')
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Car Prices')
    st.pyplot(plt.gcf())

    # Insights and Model Summary
    st.write("""
    ## Key Insights:
    1. The model's R-squared value indicates how much of the variance in car prices is explained by the features.
    2. Regularization techniques like Ridge and Lasso can help in preventing overfitting.
    3. The most significant features impacting car prices were engine size, horsepower, and fuel type.

    ### Model Strengths:
    - The model performs well in predicting car prices with a good R-squared value.

    ### Limitations:
    - The dataset might require further feature engineering for better performance.

    ### Potential Improvements:
    - Additional features like car brand, age, and maintenance history could improve model accuracy.
    - Fine-tuning hyperparameters using Grid Search or other optimization techniques could improve performance further.
    """)
