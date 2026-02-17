from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def split_data(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

def scale_amount(X_train, X_test):
    scaler = RobustScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    return X_train, X_test
