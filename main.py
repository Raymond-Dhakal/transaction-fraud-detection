from src.data_loader import load_raw_data
from src.preprocessing import split_data, scale_amount
from src.models import train_random_forest
from src.evaluation import evaluate_model
from src.__init__ import TEST_SIZE, RANDOM_STATE, N_ESTIMATORS


def main():
    # Load data
    df = load_raw_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X, y, TEST_SIZE, RANDOM_STATE
    )

    # Drop Time
    X_train = X_train.drop("Time", axis=1)
    X_test = X_test.drop("Time", axis=1)

    # Scale Amount
    X_train, X_test = scale_amount(X_train, X_test)

    # Train model
    model = train_random_forest(
        X_train, y_train, N_ESTIMATORS, RANDOM_STATE
    )

    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
