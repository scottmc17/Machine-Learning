import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier()

#DF 
animals = pd.read_csv('animal_classes.csv', usecols = ['Class_Number', 'Class_Type'])
train = pd.read_csv('animals_train.csv')
test = pd.read_csv("animals_test.csv")

train_DF = train.iloc[:,:16]
target_DF = train.iloc[:,16]
test_DF = test.iloc[:,1:]
animals_DF = test.iloc[:,0]
types_DF = animals.iloc[:,1]

knn.fit(X=train_DF, y=target_DF)

predicted = knn.predict(X=test_DF)

loc = 0
outfile = open('predictions.csv', 'w')
header = 'Animal Name, Predicted Type'
outfile.write(header + '\n')

for p in predicted:
    line = f"{animals_DF[loc]},{types_DF[int(p)-1]}\n"
    print(line)
    outfile.write(line)
    loc += 1
