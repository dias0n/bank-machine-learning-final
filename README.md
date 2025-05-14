# 💳 Bank Users Analysis with Streamlit

## 📌 Project Overview

This project demonstrates a data analysis application using **Streamlit** and **machine learning** to explore a bank's marketing data. The goal is to analyze customer information and predict whether they will subscribe to a term deposit, using various machine learning models (Random Forest, Decision Tree, and K-means Clustering).

The app allows users to upload their own dataset and interact with different preprocessing and modeling options, making it an accessible tool for analyzing similar banking datasets.

---

## 🧑‍💻 Features

### 1. Data Preprocessing:
- **Remove duplicates** and handle missing values
- **Label encoding** for categorical variables
- **MinMax Scaling** for numerical features to bring them to a common scale

### 2. Machine Learning Models:
The app supports three different models:
- **Random Forest Classifier** – for supervised learning
- **Decision Tree Classifier** – for supervised learning with model visualization
- **K-means Clustering** – for unsupervised learning, with visual clustering using PCA

### 3. Interactive Interface:
- **File Upload**: Users can upload their own CSV dataset
- **Model Selection**: Choose between Random Forest, Decision Tree, or K-means Clustering
- **Feature Selection**: Manually select which features to include in the model
- **Data Visualization**: Includes scatter plots and PCA visualization for clustering results

---

## 🧩 Data Processing & Modeling Steps

1. **Upload and Display Data:**
   - The user uploads a CSV file, which is displayed on the app.
   - Duplicate rows and missing values are removed to clean the dataset.

2. **Preprocessing:**
   - Categorical features are encoded using **LabelEncoder**.
   - Numerical features are scaled using **MinMaxScaler** to normalize them between 0 and 1.

3. **Model Selection and Training:**
   - **Random Forest** and **Decision Tree** models are trained with a selected target column and feature columns.
   - **K-means Clustering** is applied to find patterns in the data, followed by PCA for visualization in 2D space.

4. **Model Evaluation:**
   - Evaluation metrics include **accuracy**, **train/test split**, and **visualization of confusion matrix** (for classification models).

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** – for creating the web app
- **Pandas** – for data manipulation
- **NumPy** – for numerical operations
- **Scikit-learn** – for machine learning models, preprocessing, and metrics
  - Random Forest, Decision Tree, KMeans
  - LabelEncoder, MinMaxScaler
  - Accuracy, Confusion Matrix, Precision, Recall
- **Matplotlib** and **Seaborn** – for data visualization

---

## 📋 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/dias0n/bank-marketing-analysis.git
   cd bank-marketing-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📌 Sample Features and Dataset
### The app expects the uploaded CSV file to have the following features:

age — Age of the customer

job — Type of job (e.g., 'admin', 'technician', 'student', etc.)

marital — Marital status (e.g., 'single', 'married', etc.)

education — Level of education (e.g., 'primary', 'secondary', 'tertiary')

default — Whether the customer has credit in default (yes/no)

balance — Account balance in EUR

housing — Whether the customer has a housing loan (yes/no)

loan — Whether the customer has a personal loan (yes/no)

contact — Type of communication used to contact the customer (e.g., 'cellular', 'telephone', etc.)

day — Last contact day of the month

month — Last contact month of the year

duration — Duration of the last contact in seconds

campaign — Number of contacts performed during this campaign

pdays — Number of days since the customer was last contacted

previous — Number of contacts performed before this campaign

poutcome — Outcome of the previous marketing campaign (e.g., 'failure', 'success', etc.)

deposit — Target variable (whether the customer subscribed to a term deposit, yes/no)

## 📝 Notes
Make sure your dataset is clean and contains all the necessary columns for accurate model training.

The app can be used with any CSV file that follows a similar format to the one used in the project.
