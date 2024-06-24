import pandas as pd
import logistic_regression_finished

# Load the data
data1 = pd.read_csv('pseudodata_præoperation.csv')
data2 = pd.read_csv('pseudodata_præoperation.csv')
data3 = pd.read_csv('pseudodata_præoperation.csv')
data4 = pd.read_csv('pseudodata_præoperation.csv')
data5 = pd.read_csv('pseudodata_præoperation.csv')
data6 = pd.read_csv('pseudodata_præoperation.csv')

base = data1
phase1 = pd.concat([data1, data2], axis = 1)
phase2 = pd.concat([data1, data2, data3], axis = 1)
phase3 = pd.concat([data1, data2, data3, data4], axis = 1)
phase4 = pd.concat([data1, data2, data3, data4, data5], axis = 1)
phase5 = pd.concat([data1, data2, data3, data4, data5, data6], axis = 1)


data_list = [base, phase1, phase2, phase3, phase4, phase5]
for i in range(len(data_list)):
    data = data_list[i]
    logistic_regression_finished.run(data, i)
