import pandas as pd
import numpy as np
from sklearn import linear_model

# raw data from train.csv
url = 'https://raw.githubusercontent.com/egersack/PS2HouseDataExercise/master/train.csv'
data = pd.read_csv(url)

# only using the numerical data
valid_numbers = data.select_dtypes(include=[np.number])
# get rid of NaN values
valid_numbers = valid_numbers.dropna(axis=1)#is axis 0 or 1? come back

#getting the first 1000 
first_numbers = valid_numbers.head(1000)

# finding correlated data
corr = first_numbers.corr()['SalePrice'].sort_values(ascending = False)

# make X and Y values that I can use for linear regression
Y = first_numbers ['SalePrice']
X = first_numbers.drop(['SalePrice'], axis = 1)

# create linear regression, imported above
lr = linear_model.LinearRegression()
model = lr.fit(X, Y)

##predicting test data, should create a plot of the real sale price against the predicted sale price
#predictions = model.predict(X)
#plt.scatter(predictions, Y, color = 'r')

# test prediction on last part of the training data
last_numbers = valid_numbers.tail(100)
training_testing = last_numbers.drop(['SalePrice'], axis = 1)
# use model on train_test
training_prediction = model.predict(training_testing)

## scatterplot 
#plt.scatter(last_numbers['SalePrice'], training_prediction, color = 'r')

# url from test data and select only numeric data
url = 'https://raw.githubusercontent.com/egersack/PS2HouseDataExercise/master/test.csv'
test_data = pd.read_csv(url)
test_numbers =  test_data.select_dtypes(include=[np.number])
test_numbers = test_numbers.dropna(axis=1)

# create list of predicted sale prices, then round the values to two decimal places
list_predicted_prices = model.predict(test_numbers)
list_predict_rounded = np.around(list_predicted_prices, decimals=2)#np.around "evenly round to the given number of decimals"
# add predicted price to test table
test_data['SalePrice'] = list_predict_rounded

#how do i get my final data? i need it in a csv
finalOutput = test_data[['Id', 'SalePrice']]
#print predicted price and Id for user
print(finalOutput.to_string(index=False))

#export final output to a .csv
finalOutput.to_csv (r'predictions.csv', index=False, header=True)

#i do not know how to get a separate file out of this. I am not really sure what the last few lines of code do because when i put it on google colab, it returned something but here it does nothing. On my vs code othing is returned either.
