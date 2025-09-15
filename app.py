import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

# Set a title for the app
st.title('Tesla Stock Price Direction Prediction')

# --- DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_data():
    """
    This function loads the Tesla stock data, performs preprocessing,
    and engineers new features. It's cached to improve performance.
    """
    # Load the dataset
    df = pd.read_csv('Tesla.csv')
    
    # Drop the 'Adj Close' column as it is highly correlated with 'Close'
    df = df.drop(['Adj Close'], axis=1)

    # Feature Engineering from Date
    splitted = df['Date'].str.split('/', expand=True)
    df['day'] = splitted[1].astype('int')
    df['month'] = splitted[0].astype('int')
    df['year'] = splitted[2].astype('int')

    # Add a binary feature for whether the date is a quarter end
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    
    # Engineer features based on price differences
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']

    # Create the target variable: 1 if the next day's close is higher, 0 otherwise
    # This frames the problem as a classification task (predicting direction)
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # The last row will have a NaN target, so we drop it
    df = df.dropna()

    return df

# Load the data using the function
df = load_data()

# --- SIDEBAR FOR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Overview & EDA", "Model Training & Prediction"))

# --- PAGE 1: OVERVIEW & EXPLORATORY DATA ANALYSIS (EDA) ---
if page == "Overview & EDA":
    st.header("Exploratory Data Analysis")

    # Display the raw data
    st.subheader("Raw Data")
    st.write("Here are the first few rows of the Tesla stock dataset:")
    st.dataframe(df.head())

    # Check for null values
    st.write("Checking for any missing values in the dataset:")
    st.write(df.isnull().sum())
    st.success("No missing values found.")

    # Main plot: Closing price over time
    st.subheader("Tesla's Closing Stock Price Over Time")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Close'])
    ax.set_title('Tesla Close price.', fontsize=15)
    ax.set_ylabel('Price in dollars.')
    st.pyplot(fig)

    # Expander for more detailed EDA plots
    with st.expander("See More Detailed Visualizations"):
        # Feature Distribution Plots
        st.subheader("Feature Distribution")
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten() # Flatten the 2x3 grid to a 1D array
        for i, col in enumerate(features):
            sb.histplot(df[col], ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {col}')
        # Hide the empty subplot
        if len(features) < len(axes):
            axes[len(features)].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Target variable distribution
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots()
        df['target'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Price Down/Same (0)', 'Price Up (1)'])
        ax.set_title("Distribution of Target Variable (Next Day's Price Direction)")
        ax.set_ylabel('') # Hide the 'target' label on the y-axis
        st.pyplot(fig)
        st.write("The classes are well-balanced, which is good for training a classification model.")

        # Correlation Heatmap
        st.subheader("Correlation Heatmap of Highly Correlated Features")
        fig, ax = plt.subplots(figsize=(10, 8))
        # We check for correlations > 0.9 to see the highly related features
        sb.heatmap(df.drop(['Date'], axis=1).corr() > 0.9, annot=True, cbar=False, ax=ax)
        st.pyplot(fig)
        st.write("The heatmap shows that 'Open', 'High', 'Low', and 'Close' prices are highly correlated with each other, as expected. The engineered features ('open-close', 'low-high') are used to avoid multicollinearity.")


# --- PAGE 2: MODEL TRAINING & PREDICTION ---
elif page == "Model Training & Prediction":
    st.header("Model Training and Prediction")
    st.write("Here, we will train different classification models to predict if the stock price will go up or down the next day.")

    # Feature and Target selection
    features = df[['open-close', 'low-high', 'is_quarter_end']]
    target = df['target']

    # Data Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-Test Split
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features_scaled, target, test_size=0.1, random_state=2022
    )
    
    st.write(f"Training data shape: `{X_train.shape}`")
    st.write(f"Validation data shape: `{X_valid.shape}`")

    # Model Selection
    st.subheader("Choose a Model to Train")
    model_name = st.selectbox(
        "Select a model:",
        ("Logistic Regression", "Support Vector Classifier (SVC)", "XGBoost Classifier")
    )
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier (SVC)": SVC(kernel='poly', probability=True),
        "XGBoost Classifier": XGBClassifier()
    }
    
    model = models[model_name]

    # Train the model
    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, Y_train)

    st.success(f"{model_name} trained successfully!")

    # Display performance metrics
    st.subheader("Model Performance")
    
    # Use columns for a better layout
    col1, col2 = st.columns(2)

    # Training Accuracy
    train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])
    col1.metric("Training AUC Score", f"{train_auc:.4f}")

    # Validation Accuracy
    valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1])
    col2.metric("Validation AUC Score", f"{valid_auc:.4f}")

    st.write("AUC (Area Under the ROC Curve) is used as the evaluation metric. It measures the model's ability to distinguish between the two classes (price up vs. down). A score closer to 1.0 is better.")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix on Validation Data")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid, ax=ax)
    st.pyplot(fig)
    st.write("""
    - **True Positives (TP):** The model correctly predicted the price would go up.
    - **True Negatives (TN):** The model correctly predicted the price would go down or stay the same.
    - **False Positives (FP):** The model predicted the price would go up, but it went down/stayed the same (Type I error).
    - **False Negatives (FN):** The model predicted the price would go down/stayed the same, but it went up (Type II error).
    """)
