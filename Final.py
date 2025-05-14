import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Title of the app
st.title("Bank Marketing Analysis with Streamlit")

# Upload file
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader("Raw Data")
        st.write(df)

        # Data Preprocessing
        st.sidebar.subheader("Data Preprocessing")

        # Drop duplicates and missing values
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # Encoding categorical columns
        label_encoder = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Normalization of numerical columns using MinMaxScaler
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Feature selection
        feature_cols = st.sidebar.multiselect("Select feature columns", df.columns)

        # Model Selection
        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "Decision Tree", "K-means Clustering"])

        # Machine Learning Models
        if model_choice == "Random Forest" or model_choice == "Decision Tree":
            # Supervised Learning (Random Forest or Decision Tree)
            if model_choice == "Random Forest":
                st.subheader("Random Forest Model")
            else:
                st.subheader("Decision Tree Model")

            # Target variable selection
            target_col = st.sidebar.selectbox("Select target column", df.columns)

            # Model training
            if st.sidebar.button(f"Train {model_choice} Model"):
                X = df[feature_cols]
                y = df[target_col]

                # Splitting data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = DecisionTreeClassifier(random_state=42)

                model.fit(X_train, y_train)

                # Model evaluation
                st.subheader(f"{model_choice} Model Evaluation")
                train_accuracy = model.score(X_train, y_train)
                test_accuracy = model.score(X_test, y_test)
                st.write("Training Accuracy:", train_accuracy)
                st.write("Test Accuracy:", test_accuracy)

        elif model_choice == "K-means Clustering":
            # Unsupervised Learning (K-means Clustering)
            st.subheader("K-means Clustering")

            # Model training
            if st.sidebar.button("Perform K-means Clustering"):
                num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df['cluster'] = kmeans.fit_predict(df[feature_cols])

                # Data visualization
                st.subheader("Clustering Visualization (PCA)")
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(df[feature_cols])
                df_pca = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])
                df_pca['cluster'] = df['cluster']
                st.write(df_pca)

                # Plotting clusters
                st.scatter_chart(df_pca, x='PCA1', y='PCA2', color='cluster')

    except ValueError as e:
        st.error("Please upload a valid CSV file.")

