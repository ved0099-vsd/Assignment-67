import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def main():

    print("----------------------------------------")
    print("Loan Approval Prediction")
    print("----------------------------------------")

    # Step 1 : Create dataset

    X = np.array([
        [25000, 600, 200000, 10000, 0],
        [40000, 700, 300000, 8000, 1],
        [60000, 750, 500000, 12000, 1],
        [20000, 550, 150000, 15000, 0],
        [80000, 800, 700000, 10000, 1],
        [35000, 650, 250000, 9000, 1],
        [18000, 500, 100000, 12000, 0],
        [90000, 850, 800000, 15000, 1],
        [30000, 580, 200000, 14000, 0],
        [70000, 780, 600000, 10000, 1]
    ])

    Y = np.array([
        0, 1, 1, 0, 1,
        1, 0, 1, 0, 1
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

    # Step 7 : Evaluate model

    print("----------------------------------------")
    print("Step 7 : Evaluate Model")
    print("----------------------------------------")

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Actual values : ", Y_test)
    print("Predicted values : ", Y_pred)
    print("Accuracy : ", accuracy)

    # Step 8 : Predict new applicant

    print("----------------------------------------")
    print("Step 8 : Predict New Applicant")
    print("----------------------------------------")

    new_applicant = np.array([
        [55000, 720, 400000, 10000, 1]
    ])

    new_applicant = scaler.transform(new_applicant)

    prediction = model.predict(new_applicant)

    print("New Applicant : [55000, 720, 400000, 10000, 1]")
    print("Prediction : ", prediction[0])

    if prediction[0] == 0:
        print("Loan Rejected")
    else:
        print("Loan Approved")


if __name__ == "__main__":
    main()