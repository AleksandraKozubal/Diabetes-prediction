import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

data = pd.read_csv("diabetes.csv")

# Select all rows and all columns except the last one
features = data.iloc[:, :-1]
features_arrays = features.to_numpy()

# Select only the "Outcome" column
outcomes = data.iloc[:, -1]
outcome_array = outcomes.to_numpy()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features_arrays, outcome_array, test_size=0.2)

# Check which model is the most accurate
# clf_KNC = KNeighborsClassifier()
# clf_KNC.fit(X_train, Y_train)

# clf_DTC = DecisionTreeClassifier()
# clf_DTC.fit(X_train, Y_train)

clf_RFC = RandomForestClassifier()
clf_RFC.fit(X_train, Y_train)

# clf_SVC = SVC(kernel='linear', C=3)
# clf_SVC.fit(X_train, Y_train)

# print("K Neighbors Classifier accuracy: {:.4f}".format(clf_KNC.score(X_test, Y_test)))
# print("Decision Tree Classifier accuracy: {:.4f}".format(clf_DTC.score(X_test, Y_test)))
# print("Random Forest Classifier accuracy: {:.4f}".format(clf_RFC.score(X_test, Y_test)))
# print("SVC accuracy: {:.4f}".format(clf_SVC.score(X_test, Y_test)))


# Check if the entered data is in the correct format
def get_float_input(prompt, unit):
    while True:
        try:
            value = float(input(f"{prompt} ({unit}): "))
            if value < 0:
                print("Invalid input. Please enter a non-negative number.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


pregnancies = get_float_input("Pregnancies", "count")
glucose = get_float_input("Glucose", "mg/dL")
bloodPressure = get_float_input("Blood Pressure", "diastolic pressure")
skinThickness = get_float_input("Skin Thickness", "mm")
insulin = get_float_input("Insulin", "mU/ml")
bmi = get_float_input("BMI", "kg/m^2")
diabetesPedigreeFunction = get_float_input("Diabetes Pedigree Function", "unit")
age = get_float_input("Age", "years")

# Convert input data
X_predict = np.array([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]])

result = clf_RFC.predict(X_predict)

if result == 1:
    print("You may have diabetes.")
else:
    print("You probably do not have diabetes.")

print("Remember that this is only a hint, not truly medical advice.")
