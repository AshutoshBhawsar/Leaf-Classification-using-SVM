
import pandas as pd
import numpy as np

from sklearn import svm

# reading dataset
# dataset has 339 rows and 14+1 columns
dataset = pd.read_csv("leaf.csv")

# shuffle dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# split into training and testing set
X = dataset.iloc[:300, 2:16]
y = dataset.iloc[:300, 0]

Xt = dataset.iloc[300:, 2:16]
yt = dataset.iloc[300:, 0]

print("Train Dataset is \n", X)
print("Train Labels are \n", y)

print("Test dataset is \n", Xt)
print("Test labels are \n", yt)

# Train the SVM classifier
clf = svm.SVC(gamma=0.5, C=200)
clf.fit(X, y)

accu = 0
total = 39
for i, row in Xt.iterrows():
	# Predict value for the given Expression
	X_in = row.values
	y_pred = clf.predict([X_in])
	#print("Prediction is : ", y_pred)
	#print("Original is : ", yt[i])
	if (y_pred==yt[i]):
		accu = accu + 1

accu = accu / total
print ("Accuracy is ", accu)

'''
# Predict value for the given Expression
X_in = np.array([0.28064,1.0849,0.75319,0.72152,0.71404,0.13686,0.078996,1.1358,0.14122,0.2183,0.045488,0.012002,0.0015154,2.4059])
y_pred = clf.predict([X_in])
print("Prediction is : ", y_pred)
'''
