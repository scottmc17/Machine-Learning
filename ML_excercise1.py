from sklearn.datasets import fetch_california_housing 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


california = fetch_california_housing()

#print(california.DESCR)

#print(california.data.shape)

#print(california.target.shape)

#print(california.feature_names)

pd.set_option('precision', 4)
pd.set_option('max_columns', 9)
pd.set_option('display.width', None)

california_df = pd.DataFrame(california.data, columns = california.feature_names)
california_df['MedHouseValue'] = pd.Series(california.target)

#print(california_df.head())


#california_df['MedHouseValue'] = [california.target_names[i] for i in california.target]


sample_df = california_df.sample(frac=0.1, random_state=17)


sns.set(font_scale=1.1)
sns.set_style('whitegrid')

grid = sns.pairplot(data=sample_df, vars=california_df.columns[:8], hue='MedHouseValue')
plt.show()
