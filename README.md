# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
Developed By: Prasannalakshmi G
Reg No: 212222240075
```
## FEATURE SCALING :
```
import pandas as pd
import numpy as np
from scipy import stats
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")
df3=pd.read_csv("/content/bmi.csv")
df4=pd.read_csv("/content/bmi.csv")
df5=pd.read_csv("/content/bmi.csv")
```
```
df1.head()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/d4cab991-4716-47db-8142-5763c6a9ff6c)

```
df.dropna()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/301ea732-df46-438c-858a-ed73172668f3)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/01184990-a4c9-4c9b-93d4-ccbf210852c8)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/6926214c-f81b-4dbd-82e1-1b0cc52ef448)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df2[['Height','Weight']]=sc.fit_transform(df2[['Height','Weight']])
df2.head(10)

```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/03953d36-7735-4695-a5ae-0701e278d878)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/49caf31d-00cf-44c8-bd49-809f91d813a5)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/add4201c-efab-4630-867e-fde06cefc561)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df5[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df5.head()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/8e7f1596-b7de-4478-a467-104396f131f1)

## FEATURE SELECTION :

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv('/content/income.csv',na_values=[" ?"])
data
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/fd0f88d7-3b6b-4dc6-ad5d-94e595e3ac16)

```
data.isnull().sum()
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/451ed736-c914-4a2b-aee2-023fbb4ce55b)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/fd11c27d-7e58-4d0e-8cb2-d7cd7a59d2fa)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/d2239e76-922e-414b-a86d-97ca52400325)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/85095e70-c888-4eb6-af93-3bee61139f82)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/978ca1a3-893f-40c6-82c6-dc90a0e6d499)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/8f64cfcb-e895-4da5-b0e4-18e4c2caba3d)

## FEATURE SELECTION METHOD IMPLEMENTATION :

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/9c95b3b9-8e64-49d9-be8b-d9f7ebd26cf9)

```
#seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/0c1d1a9d-d137-48a0-b94b-674fe8c00a61)

```
#storing the output values in y
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/a3dc1166-ab5e-4dd2-876a-bcf251388c8d)

```
x = new_data[features].values
print(x)
```

![image](https://github.com/Prasannalakshmiganesan/EXNO-4-DS/assets/118610231/21ef09fe-3e96-4e7e-a759-7e7ac71089f5)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
