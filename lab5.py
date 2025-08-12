import pandas as pd  # for load_dataing, reading, and handling tabular data 
import numpy as np   # for numerical operations, arrays, and mathematical functions

from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets

# regression metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Clustering
from sklearn.cluster import KMeans  # KMeans clustering algorithm

# clustering metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# For feature scaling (standardizing numerical values before clustering)
from sklearn.preprocessing import StandardScaler

# For creating plots and graphs
import matplotlib.pyplot as plt


def load_data(file_path):
    # Reads a CSV file, fills missing numeric values with column means, and returns the cleaned DataFrame.
    df = pd.read_csv(file_path)  # Read CSV file into a pandas DataFrame
    # Fill NaN values in numeric columns with the column mean (ignores non-numeric columns)
    df = df.fillna(df.mean(numeric_only=True))
    return df  # Return the cleaned DataFrame


def mape(y_true, y_pred):
    # Calculates the Mean Absolute Percentage Error (MAPE).
    eps = 1e-8  # Small constant to avoid division by zero in case y_true has zeros
    # Mean of the absolute percentage errors
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))


def single_feature_regression(X_train, X_test, y_train, y_test):
    # Trains a Linear Regression model using only one feature, evaluates it on train & test sets,and returns metrics.
    model = LinearRegression()  # Create Linear Regression model instance
    model.fit(X_train, y_train)  # Train model on training data

    # Predictions for both training and test data
    y_train_pred = model.predict(X_train)  # Predict for training set
    y_test_pred = model.predict(X_test)    # Predict for test set

    # Store evaluation metrics for both training and testing
    metrics = {
        "train": {
            "MSE": mean_squared_error(y_train, y_train_pred),               # Mean Squared Error
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),     # Root Mean Squared Error
            "MAE": mean_absolute_error(y_train, y_train_pred),              # Mean Absolute Error
            "MAPE": mape(y_train, y_train_pred),                            # Mean Absolute Percentage Error
            "R2": r2_score(y_train, y_train_pred)                           # R² Score
        },
        "test": {
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MAPE": mape(y_test, y_test_pred),
            "R2": r2_score(y_test, y_test_pred)
        }
    }
    return metrics  # Return the dictionary containing train/test metrics


def multi_feature_regression(X_train, X_test, y_train, y_test):
    # Trains a Linear Regression model using multiple features, evaluates it on train & test sets, and returns metrics.
    model = LinearRegression()  # Create Linear Regression model
    model.fit(X_train, y_train)  # Train the model on training data

    # Predictions for both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Store evaluation metrics for both training and testing
    metrics = {
        "train": {
            "MSE": mean_squared_error(y_train, y_train_pred),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "MAPE": mape(y_train, y_train_pred),
            "R2": r2_score(y_train, y_train_pred)
        },
        "test": {
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MAPE": mape(y_test, y_test_pred),
            "R2": r2_score(y_test, y_test_pred)
        }
    }
    return metrics

def kmeans_clustering(X_scaled, num_clusters=2):
    # Runs KMeans clustering on scaled data and returns cluster labels and centers.
    # Create KMeans model (n_init=10 means 10 different random initializations)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(X_scaled)  # Fit the model to the scaled dataset
    return kmeans.labels_, kmeans.cluster_centers_  # Return labels and cluster centers

def evaluate_clustering(X_scaled, labels):
    # Evaluates clustering performance using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    return {
        "Silhouette": silhouette_score(X_scaled, labels),              
        "Calinski_Harabasz": calinski_harabasz_score(X_scaled, labels), 
        "Davies_Bouldin": davies_bouldin_score(X_scaled, labels)        
    }

def evaluate_k_for_kmeans(X_scaled, k_range):
    # Runs KMeans for different k values and plots Silhouette, CH, and DB scores for each k.
    
    silhouette_scores, ch_scores, db_scores = [], [], []  # Lists to store scores

    # Loop through each k value in the provided range
    for k in k_range:
        labels, _ = perform_kmeans_clustering(X_scaled, num_clusters=k)  # Run KMeans
        silhouette_scores.append(silhouette_score(X_scaled, labels))     # Silhouette score
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))      # CH score
        db_scores.append(davies_bouldin_score(X_scaled, labels))         # DB index

    # Plotting all three metrics side by side
    plt.figure(figsize=(12, 4))

    # Plot Silhouette scores
    plt.subplot(1, 3, 1)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")

    # Plot Calinski-Harabasz scores
    plt.subplot(1, 3, 2)
    plt.plot(k_range, ch_scores, marker='o', color='green')
    plt.title("Calinski-Harabasz Score vs k")
    plt.xlabel("k")
    plt.ylabel("CH Score")

    # Plot Davies-Bouldin index
    plt.subplot(1, 3, 3)
    plt.plot(k_range, db_scores, marker='o', color='red')
    plt.title("Davies-Bouldin Index vs k")
    plt.xlabel("k")
    plt.ylabel("DB Index (lower=better)")

    plt.tight_layout()  # Adjust layout so subplots don't overlap
    plt.show()


def elbow_method(X_scaled, k_range):
    # Plots inertia (sum of squared distances) for different k values to find the elbow point.
    distortions = []  # List to store inertia values

    # Loop through each k value
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)  # Create KMeans instance
        km.fit(X_scaled)  # Fit to scaled data
        distortions.append(km.inertia_)  # Inertia = within-cluster sum of squares

    # Plot inertia vs k
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, distortions, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.grid(True)  # Add gridlines for easier reading
    plt.show()

    
def main():
    # load_data dataset and clean missing numeric values
    df = load_data("MC.csv")

    # A1 & A2: Single feature regression
    X_single = df[['Year_Birth']]  # Feature: only Year_Birth column
    y = df['Income']               # Target: Income column

    # Split single-feature data into train and test sets (70% train, 30% test)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y, test_size=0.3, random_state=42)

    # Train model and get metrics
    single_metrics = single_feature_regression(X_train_s, X_test_s, y_train_s, y_test_s)

    # Print results
    print("A1 & A2: Single Attribute (Year_Birth and Income)")
    print(single_metrics["train"])
    print(single_metrics["test"])

    # A3: Multi-feature regression
    # Select all numeric columns except target
    numeric_columns = df.select_dtypes(include=[np.number]).drop(columns=['Income']).columns
    X_multi = df[numeric_columns]  # Features
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.3, random_state=42)

    # Train and evaluate multi-feature model
    multi_metrics = multi_feature_regression(X_train_m, X_test_m, y_train_m, y_test_m)
    print("\nA3: Multi-Attribute Regression (Predicting Income from all other attributes)")
    print(multi_metrics["train"])
    print(multi_metrics["test"])

    # A4: KMeans clustering
    # Prepare numeric data for clustering (exclude target and ID)
    X_cluster = df.select_dtypes(include=[np.number]).drop(columns=['Income'])
    if 'ID' in X_cluster.columns:  # Remove ID column if it exists
        X_cluster = X_cluster.drop(columns=['ID'])

    # Scale data to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Perform clustering with k=2
    labels, centers = kmeans_clustering(X_scaled, num_clusters=2)
    print("\nA4: KMeans Clustering (k=2)")
    print("Cluster labels:", labels)
    print("Cluster centers (standardized features):\n", centers)

    # A5: Clustering metrics
    cluster_metrics = evaluate_clustering(X_scaled, labels)
    print("\nA5: Clustering Metrics")
    print(cluster_metrics)

    # A6: Evaluate different k values
    evaluate_k_for_kmeans(X_scaled, k_range=range(2, 11))

    # A7: Elbow method
    elbow_method(X_scaled, k_range=range(2, 20))

if __name__ == "__main__":
    main()
