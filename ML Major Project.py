Python 3.11.2 (tags/v3.11.2:878ead1, Feb  7 2023, 16:38:35) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... import numpy as np
... 
... Source = "https://raw.githubusercontent.com/subhajyoti-prusty/publicSubhajyoti/main/Dataset.csv"
... 
... Data = pd.read_csv(Source)
... 
... # Show the entire dataset 
... print(Data)
... 
... # Check the first few rows of the dataset
... print(Data.head())
... 
... # Check the data types of the columns
... print(Data.dtypes)
... 
... # Check the shape of the dataset
... print("The shape of the data: ",Data.shape)
... 
... # Check the size of the dataset
... print("The size of the data: ",Data.size)
... 
... # Check the info of the dataset
... print(Data.info())
... 
... # Slicing the dataset form row 32 to 46
... print(Data[32:46])
... 
... # Slicing the dataset form row 32 to 46 and column index 0 to 3
... print(Data.iloc[32:46,0:4])
... 
... #Check the number of unique value of the dataset
... print("The number of unique values the sex column has is",Data.Sex.nunique())
... 
... #Check the unique value of the dataset
... print("The unique values the sex column has is",Data.Sex.unique())

#Group by survived or not (survived=1 and Died=0)
print(Data.groupby('Survived').size())

# Check for missing values
print(Data.isnull().sum())

# Compute the percentage of missing values in each column
print(Data.isnull().mean() * 100)

# Check the summary statistics of the dataset
print(Data.describe())

# Remove the unnecessary columns
Data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert the Sex column to binary values (0 = male, 1 = female)
Data['Sex'] = np.where(Data['Sex'] == 'male', 0, 1)

# Fill missing Age values with the median
Data['Age'].fillna(Data['Age'].median(), inplace=True)


# Fill missing Embarked values with the mode
Data['Embarked'].fillna(Data['Embarked'].mode()[0], inplace=True)

# Convert the Embarked column to numerical values (0 = S, 1 = C, 2 = Q)
Data['Embarked'] = Data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

P = Data.drop('Survived', axis=1)
Q = Data['Survived']

from sklearn.model_selection import train_test_split

P_train, P_test, Q_train, Q_test = train_test_split(P, Q, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(P_train, Q_train)
from sklearn.metrics import accuracy_score

Q_pred = clf.predict(P_test)
accuracy = accuracy_score(Q_test, Q_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
