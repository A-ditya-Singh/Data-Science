# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Function to create sample data
def create_sample_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    return X, y

# Function to split data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to standardize data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to train linear regression model
def train_linear_regression(X_train_scaled, y_train):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

# Function to make predictions
def make_predictions(model, X_test_scaled):
    y_pred = model.predict(X_test_scaled)
    return y_pred

# Function to calculate mean squared error
def calculate_mse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Function to load Iris dataset
def load_iris_dataset():
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    return X_iris, y_iris

# Function to train KNN classifier
def train_knn(X_train_iris, y_train_iris):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_iris, y_train_iris)
    return knn

# Function to apply PCA for dimensionality reduction
def apply_pca(X_iris):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_iris)
    return X_pca

# Function to create plot
def create_plot(x, y1, y2):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label='Sine Wave', color='blue', linestyle='--')
    plt.plot(x, y2, label='Cosine Wave', color='red', linestyle='-')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sine and Cosine Waves')
    plt.legend()
    plt.grid()
    plt.show()

# Function to display image
def display_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.title('Displayed Image')
    plt.show()

# Function to create bar chart
def create_bar_chart(categories, values):
    plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart Example')
    plt.show()

# Function to create scatter plot
def create_scatter_plot(x_scatter, y_scatter):
    plt.scatter(x_scatter, y_scatter, color='orange', marker='o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot Example')
    plt.show()

# Main function
def main():
    # Create sample data
    X, y = create_sample_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Standardize data
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Train linear regression model
    model = train_linear_regression(X_train_scaled, y_train)

    # Make predictions
    y_pred = make_predictions(model, X_test_scaled)

    # Calculate mean squared error
    mse = calculate_mse(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Load Iris dataset
    X_iris, y_iris = load_iris_dataset()

    # Split Iris dataset
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = split_data(X_iris, y_iris)

    # Train KNN classifier
    knn = train_knn(X_train_iris, y_train_iris)

    # Make predictions
    y_pred_iris = knn.predict(X_test_iris)

    # Apply PCA for dimensionality reduction
    X_pca = apply_pca(X_iris)
    print("PCA Transformed Data:\n", X_pca[:5])

    # Create plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    create_plot(x, y1, y2)

    # Display random image
    random_image = np.random.rand(100, 100, 3)
    plt.imshow(random_image)
    plt.axis('off')
    plt.title('Random Image')
    plt.show()

    # Display image
    img = plt.imread('sample_image.png')
    display_image(img)

    # Create bar chart
    categories = ['A', 'B', 'C', 'D']
    values = [10, 20, 15, 25]
    create_bar_chart(categories, values)

    # Create scatter plot
    x_scatter = np.random.rand(50)
    y_scatter = np.random.rand(50)
    create_scatter_plot(x_scatter, y_scatter)

if __name__ == "__main__":
    main()