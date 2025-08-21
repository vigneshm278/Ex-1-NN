<H3>ENTER YOUR NAME : VIGNESH M</H3>
<H3>ENTER YOUR REGISTER NO : 212223240176</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 20/08/2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
### Dataset:
![images](https://github.com/user-attachments/assets/7d7aaae0-826d-4cae-bbd8-a2287d7de089)

### X Values:
![images](https://github.com/user-attachments/assets/18342c94-abef-4229-b26d-0acdb5c58626)
### Y Values:
![images](https://github.com/user-attachments/assets/52944df9-1956-4e6a-bfab-ad0f187142ed)
### Null Values:
![images](https://github.com/user-attachments/assets/42d3958f-90c9-4e02-9cac-40289956eaa0)
### Duplicated Values:
![images](https://github.com/user-attachments/assets/aa48c2ba-8d64-4d60-b941-21c8d9bd3c21)
### Description:
![images](https://github.com/user-attachments/assets/2e70d654-8a26-419f-8cb0-46b4d84a033c)
### Normalized Dataset:
![](https://github.com/user-attachments/assets/1b0ea631-5c44-476d-bee6-2a8ddcdabb98)
### Training Data:
![](https://github.com/user-attachments/assets/3c91a26c-f398-4253-a2ed-e6acac6ef78a)
### Testing Data:
![](https://github.com/user-attachments/assets/679b8a7b-057e-4bdb-ac4c-3da60c7aa258)
![image](https://github.com/user-attachments/assets/9e77edeb-3c45-4eb7-a5ca-88bea4b9418e)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


