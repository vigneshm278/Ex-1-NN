<H3>ENTER YOUR NAME : VIGNESH M</H3>
<H3>ENTER YOUR REGISTER NO : 212223240176</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 21/08/2025</H3>
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
<img width="1478" height="370" alt="image" src="https://github.com/user-attachments/assets/86056e11-d4bd-4c45-bf4b-6d96d9b01323" />

### X Values:
<img width="916" height="268" alt="image" src="https://github.com/user-attachments/assets/a7f94898-8332-45a9-a68f-eddc77bb5310" />

### Y Values:
<img width="513" height="125" alt="image" src="https://github.com/user-attachments/assets/19cc51f1-c355-47d6-944f-b26e29c4bdf8" />

### Null Values:
<img width="426" height="706" alt="image" src="https://github.com/user-attachments/assets/4b168397-8adc-4a4d-8079-fd899d8ebb75" />

### Duplicated Values:
<img width="449" height="633" alt="image" src="https://github.com/user-attachments/assets/6af08908-bf2b-4e33-b360-98b38d69b2e8" />

### Description:
<img width="1483" height="485" alt="image" src="https://github.com/user-attachments/assets/59e266a4-4dae-472a-aaca-baaa8844f637" />

### Normalized Dataset:
<img width="1388" height="359" alt="image" src="https://github.com/user-attachments/assets/f7f93b33-da7f-44c6-b027-002d502b7fd2" />

### Training Data:
<img width="915" height="696" alt="image" src="https://github.com/user-attachments/assets/4b0c88d4-9979-4064-967b-7a57a092eb00" />

### Testing Data:
<img width="823" height="296" alt="image" src="https://github.com/user-attachments/assets/2b97658f-85b5-4b93-96cd-5e7854a4bab2" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


