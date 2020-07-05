"""
boosting algorithm #1: xgboost using tree base learner

~~~~~~~~~~~~~~~~~~~~~~~~~

"""
# Import xgboost
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# get data in
data = pd.read_csv('Concession_Date_Converted_df.csv')
data = data.drop(columns=['TimeSlice_Start_Time', 'Item_Count'])

# lost 1200 rows of data due to no gross amount
data = data[data['Gross_Amount'] != 0]

# split data, data is split roughly on the first 23 months and last month
split = round(23.95/24*(len(data)))

#split = round(23.7/24*(len(data)))

# Create the training and test sets
concession_train = data[:split]

concession_test = data[split:]

X_train, y_train = concession_train.iloc[:,
                                         :-1], concession_train.iloc[:, -1]
X_test, y_test = concession_test.iloc[:,
                                      :-1], concession_test.iloc[:, -1]


# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(n_estimators=500)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train, eval_set=[
           (X_train, y_train), (X_test, y_test)], early_stopping_rounds=100, verbose=True)

# Feature Importances: Which features the model is relying on most to make the prediction. Sums up how many times each feature is split on.
important_features = plot_importance(xg_reg, height=0.9)


# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)
# append prediction column to test data side by side
# prediction_df = pd.DataFrame(xg_reg.predict(X_test), columns=["Prediction"])
# concession_test["Prediction"] = prediction_df

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# Compute the mae: mae
mae = mean_absolute_error(y_test, preds)
print("MAE: %f" % (mae))

# Compute the mape: mape
mape = mean_absolute_percentage_error(y_true=y_test, y_pred=preds)
print("MAPE: %f" % (mape))


# plot
# plot real sale vs prediction to compare


fig = plt.figure()
ax1 = fig.add_subplot(111)

compare_pred_real = {}
compare_pred_real["prediction"] = preds
compare_pred_real["real"] = y_test
compare_pred_real = pd.DataFrame(compare_pred_real, columns=[
                                 'prediction', 'real']).reset_index()


compare_pred_real.reset_index().scatter(x='index', y='prediction')
compare_pred_real.reset_index().scatter(x='index', y='real', c='r')
plt.title('Scatter Plot Comparison for Real Gross_Amount vs Prediction')
plt.xlabel('Progression in time')
plt.ylabel('$')
plt.legend()
plt.show()
