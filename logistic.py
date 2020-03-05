import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Comment out whichever (binary or multiclass) type you don't want to run at the end of the file

def binaryClassification():
    print('=============================')
    print('=== Binary Classification ===')
    print('=============================')
    print('* Uses the sampleLogisticData.csv file of dummy data\n')
    dataset = pd.read_csv('./sampleLogisticData.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # can also use train_test_split(dataset[['age']], dataset.bought_insurance, ...) instead
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    model = LogisticRegression()
    # fits the model using the training data
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print('Score:', score)

def multiclassClassification():
    print('=================================')
    print('=== Multiclass Classification ===')
    print('=================================')
    print('* Uses the load_digits from sklearn - see sklearn\'s documentation for more info\n')
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print('Score:', score)
    
    # Confusion matrix to help see the accuracy of the model
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    import seaborn as sn
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

binaryClassification()
multiclassClassification()