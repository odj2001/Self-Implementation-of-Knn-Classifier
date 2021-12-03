import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from scipy.spatial import distance

def DataDescription(x, y, digits, testSplit ):
    """This function will print out information about a given dataset"""
    print("\nDATASET INFORMATION\n")
    print("Number of data entries:", len(x))
    print("Number of classes:", len(digits.target_names))
    print("Number of features in each data entry:", len(digits.feature_names))
    
    print("The data has been split:")
    print(testSplit*100, "% Test Data")
    print((1-testSplit)*100, "% Train Data")
    
    """Sets the min and max values to the first value in the dataset, allowing it to be compared"""
    minValue = x[0][0]
    maxValue = x[0][0] 
    """Loops through the each feature in the array in order to find the max and min values """
    for i in range(len(x)):
        for j in range(len(digits.feature_names)):
            """minValue and maxValue are updated to the new min and max value whenever they are smaller or larger respectively"""
            if (x[i][j]< minValue):
                minValue = x[i][j]
            if (x[i][j]> maxValue):
                maxValue = x[i][j]
                
    print("Minimum value in each feature:",minValue)
    print("Maximum value in each feature:",maxValue,"\n")
    
    dataEntriesPerClass = []
    """The nested for loop runs through each class and creates an array with the number of times they appear in the target array"""
    for i in range(len(digits.target_names)):  
        count = 0
        for j in range(len(y)):
            """Checks if the value of the dataset and the classes are equal and adds to count if this is true"""
            if y[j] == digits.target_names[i]:
                count+=1
        """Appends the number of entries to a given class onto an array"""
        dataEntriesPerClass.append(count)
              
    for i in range (len(digits.target_names)):
        print("The class", digits.target_names[i], "has", dataEntriesPerClass[i], "data entries" )     
    
    print("\n***************************************")
   

def F2KNearestNeighbours(x_train, y_train, x_test, k):
    """This will return the prected values of the given x_train values"""
    knn = neighbors.KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    return (knn.predict(x_test))

def F3KNearestNeighbours(x_train, y_train, x_test, k):
    predictedValue = []
    
    """The nested for loops will be used to find the euclidean distance of each x_train entry from every X_test entry"""
    for i in range(len(x_test)):
        eucDistances = []
        """Turns each data entry into a numpy array so the euclidean distance can be calculated"""
        for j in range(len(x_train)):
            """The euclidean distance is calculated b"""
            eucDistance = distance.euclidean(x_test[i],x_train[j])
            eucDistances.append([eucDistance, y_train[j]])
                      
        eucDistances.sort(key=lambda tup: tup[0])
        #eucDistances = sorted(eucDistances, key = itemgetter(0))
        eucDistances = eucDistances[:k]
        orderedKValues = [x[1] for x in (eucDistances[:k])]
                  
        predictedValue.append (np.bincount(orderedKValues).argmax())
         
    return predictedValue



def AccuracyTest(predictedValues, y_test):
    """This function will print out the accuracy of the values predicted by the K-nn function compared to the actual values"""
    errorCount = 0
    numValues = len(predictedValues)
    """
    Loops fo reach value in predictedValues and compares it to the actual value in y_test. 
    If these are no the same this will add to the error count
    """
    for i in range(numValues):
        if predictedValues[i] != y_test[i]:
            errorCount += 1
    """Outputs the number of correct values, incorrect values and the accuracy as a percentage"""
    print ("Correct Values:", (numValues - errorCount))
    print ("Incorrect Values:", errorCount)
    print ("Accuracy:", (100 - ((errorCount / numValues * 100))),"%")  
    print("\n***************************************")
    





"Loads the data set and splits it into the data and the target values that should be expected"    
digits = load_digits()
x = digits.data
y = digits.target

"""Takes the users input for the test split percentage and the number of neighbours that will be considered"""
testSplit = input("Please enter a number between 0-1 for the percentage of testing data ")
testSplit = float(testSplit)
k = input("Please enter the number of neighours ")
k = int(k)
print("\n***************************************")

"""This splits the data into train data and test data. The split is decided by the number given in test_size"""
DataDescription(x,y, digits, testSplit)

"""This splits the data into train data and test data. The split is decided by the number given in test_size"""
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = testSplit)

"""Runs the KNearestNeighbours function in order to create testPredictions and train Predictions"""
f2TestPredictions = F2KNearestNeighbours(x_train,y_train,x_test, k)
f2TrainPredictions = F2KNearestNeighbours(x_train,y_train,x_train, k)

f3TestPredictions = F3KNearestNeighbours(x_train,y_train,x_test, k)
f3TrainPredictions = F3KNearestNeighbours(x_train,y_train,x_train, k)

print("---ACCURACY TESTS ON THE SKLEARN IMPLEMENTATION---")
print ("TEST DATA ACCURACY")
AccuracyTest(f2TestPredictions, y_test)
print ("TRAIN DATA ACCURACY")
AccuracyTest(f2TrainPredictions, y_train)
print("\n***************************************")       
print("---ACCURACY TESTS ON THE SELF IMPLEMENTATION---")
print ("TEST DATA ACCURACY")
AccuracyTest(f3TestPredictions, y_test)
print ("TRAIN DATA ACCURACY")
AccuracyTest(f3TrainPredictions, y_train)
       
    
