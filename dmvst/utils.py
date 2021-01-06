import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from math import sqrt

# def normalization(data, test_ratio):
# 	tr_data = data[:-int(len(data)*test_ratio)*2]
# 	print(tr_data.max(), tr_data.min())
# 	tr_data = tr_data / tr_data.max()
# 	return tr_data, tr_data.max()


# def smoothing(vec):
# 	for i in range(len(vec)):
# 		vec[i] = 0.00001 if vec[i] == 0 else vec[i]
# 	return vec


def root_mean_squared_error(y_true, y_pred):
	return sqrt(metrics.mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred): 
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	# y_true = smoothing(y_true)
	# return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

	mae, num = 0.0, 0
	for i in range(len(y_true)):
		if y_true[i] != 0:
			num += 1
			mae += np.abs((y_true[i] - y_pred[i]) / (y_true[i]))
	mape = (mae / num) * 100
	return mape