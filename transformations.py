import numpy
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def feature_standardization(X):
    """
    Performs feature standardization on the input feature matrix X.

    Args:
        X (numpy.ndarray): Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.

    Returns:
        numpy.ndarray: Standardized feature matrix of shape (m, n).

    """

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    X_standardized = scaler.fit_transform(X)

    return X_standardized

def feature_normalization(X):
    """
    Performs feature normalization on the input feature matrix X to the range [-1, 1].

    Args:
        X (numpy.ndarray): Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.

    Returns:
        numpy.ndarray: Normalized feature matrix of shape (m, n).

    """

    # Create a MinMaxScaler object with feature range set to (-1, 1)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    X_normalized = scaler.fit_transform(X)

    return X_normalized