import pandas
import numpy
# Import the SVM Classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Read the data
data = pandas.read_csv('svm_data.csv')

# Split the data into X and y
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


classifier = SVC(kernel = 'rbf', gamma = 200)

# Fit the classifier
classifier.fit(X_train,y_train)
