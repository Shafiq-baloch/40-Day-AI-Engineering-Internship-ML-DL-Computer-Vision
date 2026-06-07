#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load the iris dataset
iris = load_iris()

X = iris.data
y = iris.target

#print dataset information
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Classes:")
print(iris.target_names)

#split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#train linear SVM classifier
linear_svm = SVC(kernel='linear')

linear_svm.fit(X_train, y_train)

#predict on the test set
linear_pred = linear_svm.predict(X_test)

#calculate accuracy
linear_accuracy = accuracy_score(y_test, linear_pred)
print("\nLinear SVM Accuracy:", linear_accuracy)

#train RBF SVM classifier
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)

#predict on the test set
rbf_pred = rbf_svm.predict(X_test)

#calculate accuracy
rbf_accuracy = accuracy_score(y_test, rbf_pred)
print("\nRBF SVM Accuracy:", rbf_accuracy)

#compare accuracies
print("\nModel Comparison")
print("-------------------")
print("Linear SVM:", linear_accuracy)
print("RBF SVM   :", rbf_accuracy)


#adding polynomial kernel
poly_svm = SVC(
    kernel='poly',
    degree=3
)

poly_svm.fit(X_train, y_train)

poly_pred = poly_svm.predict(X_test)

poly_accuracy = accuracy_score(
    y_test,
    poly_pred
)

print("Polynomial SVM:",
      poly_accuracy)


#now comparing all three models
print("\nKernel Comparison")
print("-------------------")
print("Linear     :", linear_accuracy)
print("RBF        :", rbf_accuracy)
print("Polynomial :", poly_accuracy)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#using only two features of iris
iris = load_iris()

X = iris.data[:, :2]
y = iris.target

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#train linear SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

#create a mesh grid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

#predict on the mesh grid
Z = svm.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)


#plotting
plt.figure(figsize=(10,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    edgecolors='k'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.title("Linear SVM Decision Boundary")

plt.savefig("graphs/linear_svm_boundary.png")

plt.show()

#TASK 2: SMS Spam Classifier (TF-IDF + Naive Bayes)

#loading the dataset
import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")
df.head()

#cleaning the dataset
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

#converting labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

#tf-idf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#training Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

#evaluating the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))