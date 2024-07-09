import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


file_path = './dataset/qt_dataset.csv'
data = pd.read_csv(file_path)

data = data.drop(columns=['Temperature'])

data['Result'] = data['Result'].map({'Negative': 0, 'Positive': 1})

X = data[['Oxygen', 'PulseRate']]
y = data['Result'] #Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

# svm_model = SVC(kernel='linear', random_state=42)
# svm_model.fit(X_train, y_train)

# svm_y_pred = svm_model.predict(X_test)
# svm_accuracy = accuracy_score(y_test, svm_y_pred)
# svm_report = classification_report(y_test, svm_y_pred)

print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Random Forest Classification Report:\n{rf_report}")

# print(f"SVM Accuracy: {svm_accuracy}")
# print(f"SVM Classification Report:\n{svm_report}")
