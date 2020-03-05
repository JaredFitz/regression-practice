import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./sampleLinearData.csv')

# First let's test for a simple linear regression of a made up sample set of dummy data
X = dataset.iloc[:, :-1].values
# Assumes the last column is the dependent variable
y = dataset.iloc[:, -1].values

##################################################
# # Encoding Categorical Data - would be needed if some of the columns were strings
# # and needed to be turned into numbers
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features=[3])
# X = onehotencoder.fit_transform(X).toarray()
# # Avoiding the Dummy Variable
# X = X[:, 1:]
##################################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Check R2 values
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('Coefficients: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)
print('R2 Value: ', r2)

# Plot the test points & regression line if there is only one independent variable
if len(X[0]) == 1:
    plt.scatter(X, y, color='gray')
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.legend(['y = ' + str(round(regressor.coef_[0], 2)) + 'x + ' + str(round(regressor.intercept_, 2)), 'All Data', 'Test Data'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dummy Data Regression')
    plt.show()



