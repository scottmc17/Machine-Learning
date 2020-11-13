from sklearn.datasets import load_digits

digits = load_digits() #returns a bunch object.

#print(digits.DESCR)

#print(digits.data[:2]) #this represents the pixels 

#print(digits.data.shape)

#print(digits.target[:2]) # this represents the target which is the number 

#print(digits.target.shape)

#print(digits.images[:2]) #we need this array to feed to our model

import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split

fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))

#python zip function bundels the 3 itertables and produces one iterable
for item in zip(axes.ravel(), digits.images, digits.target):
    axes,image,target = item 
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()

#plt.show()


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 11)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier #KN is our model 

knn = KNeighborsClassifier()

# load the training data into the model using the fit model  #this is the ML part 
knn.fit(X=x_train, y=y_train)

predicted = knn.predict(X=x_test) # we don't give it a y bc we want it to find the y

expected = y_test 

print(predicted[:20])

print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e] # zip iterated through both at the same time 

print(wrong) # gives a list of what it guessed to what it actually was

print(format(knn.score(x_test, y_test), ".2%")) #shows how accurate the model is 

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_true=expected, y_pred=predicted)

print(cf)

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt2

cf_df = pd.DataFrame(cf, index=range(10), columns=range(10))

fig = plt2.figure(figsize=(7,6))
axes = sns.heatmap(cf_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt2.show()