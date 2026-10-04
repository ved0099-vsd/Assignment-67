import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def main():

    print("----------------------------------------")
    print("Customer Churn Prediction")
    print("----------------------------------------")

    # Step 1 : Create dataset

    X = np.array([
        [25, 500, 12, 1, 2],
        [30, 700, 24, 0, 1],
        [45, 1200, 6, 5, 8],
        [50, 1500, 5, 6, 10],
        [28, 600, 18, 1, 1],
        [35, 800, 10, 2, 3],
        [48, 1400, 4, 7, 9],
        [52, 1600, 3, 8, 12],
        [27, 550, 20, 0, 1],
        [42, 1300, 8, 4, 7]
    ])

    Y = np.array([
        0, 1, 1, 0, 1,
        0, 1, 1, 0, 1
    ])

    print("Dataset created successfully")

    # Step 2 : Display dataset information

    print("----------------------------------------")
    print("Step 2 : Dataset Information")
    print("----------------------------------------")

    print("X shape : ", X.shape)
    print("Y shape : ", Y.shape)

    # Step 3 : Split dataset

    print("----------------------------------------")
    print("Step 3 : Split Dataset")
    print("----------------------------------------")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Training data : ", X_train.shape)
    print("Testing data : ", X_test.shape)

    # Step 4 : Apply StandardScaler

    print("----------------------------------------")
    print("Step 4 : Feature Scaling")
    print("----------------------------------------")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Feature scaling completed")

    # Step 5 : Create FNN model

    print("----------------------------------------")
    print("Step 5 : Create FNN Model")
    print("----------------------------------------")

    model = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    # Step 6 : Train model

    print("----------------------------------------")
    print("Step 6 : Train Model")
    print("----------------------------------------")

    model = model.fit(X_train, Y_train)

    print("Model trained successfully")

    # Step 7 : Evaluate accuracy

    print("----------------------------------------")
    print("Step 7 : Evaluate Model")
    print("----------------------------------------")

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Actual values : ", Y_test)
    print("Predicted values : ", Y_pred)
    print("Accuracy : ", accuracy)

    # Step 8 : Predict new customer

    print("----------------------------------------")
    print("Step 8 : Predict New Customer")
    print("----------------------------------------")

    new_customer = np.array([
        [46, 1450, 5, 6, 9]
    ])

    new_customer = scaler.transform(new_customer)

    prediction = model.predict(new_customer)

    print("New Customer : [46, 1450, 5, 6, 9]")
    print("Prediction : ", prediction[0])

    if prediction[0] == 0:
        print("Customer will stay")
    else:
        print("Customer may leave")


if __name__ == "__main__":
    main()