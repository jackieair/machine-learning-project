#import libraries needed for the project
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import visuals as vs

get_ipython().run_line_magic('matplotlib', 'inline')

#load the dataset
data = pd.read_csv('housing.csv')
#show the first 6 row
print(data.head(6))

#split the data into results(prices) and features
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape)) #星号解析参数成位置参数传入



minimum_price = prices.min()
maximum_price = prices.max()
mean_price = prices.mean()
median_price = prices.median()
std_price = prices.std()

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:.2f}".format(minimum_price)) 
print("Maximum price: ${:.2f}".format(maximum_price))
print("Mean price: ${:.2f}".format(mean_price))
print("Median price ${:.2f}".format(median_price))
print("Standard deviation of prices: ${:.2f}".format(std_price))

#设定性能评估函数
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    #R^2表示预测值与实际值相关程度平方的百分比
    score = r2_score(y_true, y_predict)
    
    return score


#split the data into train and test set, import train_test_split
from sklearn.model_selection import train_test_split

#random_state 设置后每次分割数据都相同，否则每次都不同
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 1) 

print("Training and testing split was successful.")




# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)

vs.ModelComplexity(X_train, y_train)


params = {"max_depth":list(range(1,11))}

#main function
#Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,KFold

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42) 
    #比价下与K折交叉验证区别, cross_validator = KFold(n_splits=10)

    regressor = DecisionTreeRegressor(random_state = 1)

    params = {'max_depth': list(range(1,11))}

    scoring_fnc = make_scorer(performance_metric)

    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# 为客户数据创建输入矩阵
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

#显示预测结果
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


r2 = performance_metric(y_test, reg.predict(X_test))
print("Optimal model has R^2 score {:.2f} on test data".format(r2))
