import pandas as  pd
from sklearn.model_selection import train_test_split

#Change file name
#Change Temp to Value.

nyc = pd.read_csv('ave_yearly_temp_nyc_1895-2017.csv')
#nyc.columns = ['Date', 'Temperature', 'Anomaly']


#rint(nyc.head(3)) # this displays the 1st 3 samples

#print(nyc.Date.values)

#print(nyc.Date.values.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(            # x is the data # y is the target #line 18 is the issue 
    nyc.Date.values.reshape(-1,1), nyc.Value.values,
    random_state=11)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

print(lr.coef_)
print(lr.intercept_)

predicted = lr.predict(x_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e: .2f}")

predict = (lambda x: lr.coef_ * x + lr.intercept_)

print(predict(2020))

print(predict(1890))

print(predict(2021))

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x='Date',
    y='Value',
    hue='Value',
    palette='winter',
    legend=False
)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)

y = predict(x)

print(y)

import matplotlib.pyplot as plt
line = plt.plot(x,y)
plt.show()

# the 1st linear regression was for January ( tracks all jan temps)
# the 2nd linear regression was for the end of the year avg temp. As the year goes on 
# the temp increases. 
# 
# (the '12' and the end of each year is show that it's for the whole year)