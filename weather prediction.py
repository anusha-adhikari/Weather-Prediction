import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sea
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC


data=pd.read_csv('/Users/macuser/Desktop/python + ML/weather prediction/seattle-weather.csv')

#printing 10 random rows as sample
print(data.sample(10), end = '\n\n')

# printing the data description and dataset information
print(data.describe(), end = '\n\n')
print(data.info(), end = '\n\n')

#checking if there are any null values in the dataset
print(data.isnull().sum(), end = '\n\n')

#count no.of unique elements in the columns
print(data.nunique(), end = '\n\n')

#plotting the graphs
plt.figure(figsize=(18,6))
sea.pairplot(data.drop('date',axis=1),hue='weather')
plt.show()

#making a dictionary of the weather conditions
weather_con = {'drizzle': 0, 'fog': 1, 'rain': 2, 'snow': 3, 'sun': 4}
weather_dec = {0: 'drizzle', 1: 'fog', 2:'rain', 3: 'snow', 4: 'sun'}

#dropping the date column
dataset = data.drop('date', axis = 1)

#converting weather to type category
dataset['weather'] = dataset['weather'].astype('category')

#categorising variables of weather into numbers
dataset['weather'] = dataset['weather'].cat.codes

#preparing X and Y for training the model
X = dataset.drop('weather', axis = 1)
Y = dataset['weather']

#Splitting the dataset for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Cleansing the data
sc = StandardScaler() #used for normalization
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
#Logistic Regression
log=LogisticRegression(random_state=0)
log.fit(X_train,Y_train)
print('Test accuracy of Logistic Regression: {}'.format(log.score(X_test, Y_test)*100))

#K Nearest Neighbors
knn=KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,Y_train)
print('Test accuracy of K Nearest Neighbors: {}'.format(knn.score(X_test, Y_test)*100))

#Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rfc.fit(X_train, Y_train,sample_weight=None)
print('Test accuracy of Random Forest: {}'.format(rfc.score(X_test, Y_test)*100))

#Decision Tree 
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(X_train, Y_train,sample_weight=None)
print('Test accuracy of Decision Tree: {}'.format(dtc.score(X_test, Y_test)*100))

#Support Vector Machine
svm = SVC(random_state = 0)
svm.fit(X_train, Y_train)
print('Test accuracy of support vector machine: {}'.format(svm.score(X_test, Y_test)*100))
'''

#Naive Bayes Algorithm - max
nba = GaussianNB()
nba.fit(X_train, Y_train)
print('Test accuracy of Naive Bayes Algorithm: {}'.format(nba.score(X_test, Y_test)*100))

y_pred = nba.predict(X_test)

decodeFunc1 = np.vectorize(lambda val : weather_dec[val])
decodeFunc2 = np.vectorize(lambda val : weather_dec[val])

decodedWeather1 = decodeFunc1(Y_test)
decodedWeather2 = decodeFunc2(y_pred)

df = np.concatenate((decodedWeather1, decodedWeather2)).reshape(-1,2)
dataframe = pd.DataFrame(df, columns=['Real Weather', 'Predicted Weather'])
print(dataframe)
