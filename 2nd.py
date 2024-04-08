import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("path/to/titanic.csv")




data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  


data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Major', 'Sir', 'Dr', 'Rev', 'Ms', 'Mlle'], 'Other', inplace=True)


data = pd.get_dummies(data, columns=['Sex', 'Pclass', 'Embarked', 'Title'])

X = data.drop('Survived', axis=1)
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logreg = LogisticRegression(solver='liblinear')  
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print('Logistic Regression Accuracy:', logreg_accuracy)

svm = SVC(kernel='linear')  
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print('SVM Accuracy:', svm_accuracy)

dtree = DecisionTreeClassifier(max_depth=3)  
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
dtree_accuracy = accuracy_score(y_test, y_pred_dtree)
print('Decision Tree Accuracy:', dtree_accuracy)

print("\nBest Performing Model based on Accuracy:")
if logreg_accuracy >= svm_accuracy and logreg_accuracy >= dtree_accuracy:
    print("Logistic Regression")
elif svm_accuracy >= logreg_accuracy and svm_accuracy >= dtree_accuracy:
    print("SVM")
else:
    print("Decision Tree")


