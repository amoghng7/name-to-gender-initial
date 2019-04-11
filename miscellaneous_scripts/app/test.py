import pandas as pd
data = pd.read_csv("new_indian_name_dataset.csv")
from preprocessing import preprocess, get_features

data = data.drop(columns=["name"])

X = data.iloc[:,:-1]
y = data.iloc[:,-1:]

X_train, X_test, y_train, y_test = preprocess(X, y)

dtypes = data.drop(columns=["gender"]).dtypes

print(dtypes)

# For Training
from sklearn.tree import DecisionTreeClassifier

# For Validating
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# For Visualization
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
import pylab as pl

# Fitting Naive Bayes to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy: {0:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("Precision Score: {0:.2f}%".format(precision_score(y_test,y_pred)*100))
print("Recall Score: {0:.2f}%".format(recall_score(y_test,y_pred)*100))
print("F1 Score: {0:f}".format(f1_score(y_test,y_pred)))

df_cm = pd.DataFrame(cm, index=['Male', 'Female'], columns=['Male', 'Female'])
# plt.figure(figsize = (10,7))
print("\nconfusion Matrix: ")
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap='Blues', fmt='g')
plt.show()

feat = get_features("aditya", dtypes)

print(feat)