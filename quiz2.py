import pandas as pd 


classes = pd.read_csv('animal_classes.csv', usecols = ['Class_Number', 'Class_Type'])
train = pd.read_csv('animals_train.csv')


df = pd.DataFrame(classes)
df_2 = pd.DataFrame(train)

#print(df_2)

#df_result = pd.DataFrame(classes, columns=['value'])


'''
for label, content in df_2.items():
    print(f'label: {label}')
    print(f'content: {content}', sep='\n')
'''

'''
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=df_2, y=df_2["Class_Number"])
'''