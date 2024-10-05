"""
    @author Alex Liao
    @email biganiseed@gmail.com
    @date Sep. 5, 2024
    @version: 0.1.0
    @description:
                This module contains common functions for reshaping the BGP features.

    This Python code (versions 3.8+)
"""

import numpy as np

"""
    According to the feature definition, apply different aggregation function to different column.

    Columns 1-4: time (column 1: hour+minute; column 2: hour; column 3: minute; column 4: second)
    Columns 5-41: features
    List of features extracted from BGP update messages:
    1. Number of announcements
    2. Number of withdrawals
    3. Number of announced NLRI prefixes
    4. Number of withdrawn NLRI prefixes
    5. Average AS-path length
    6. Maximum AS-path length
    7. Average unique AS-path length
    8. Number of duplicate announcements
    9. Number of implicit withdrawals
    10. Number of duplicate withdrawals
    11. Maximum edit distance
    12. Arrival rate [Number]
    13. Average edit distance
    14-23. Maximum AS-path length = n, where n = (11, ...,20)
    24-33. Maximum edit distance = n, where n = (7, ...,16)
    34. Number of Interior Gateway Protocol (IGP) packets
    35. Number of Exterior Gateway Protocol (EGP) packets
    36. Number of incomplete packets
    37: Packet size (B) [Average]

    This list comes from src/CSharp_Tool_BGP/Readme.md
    The last column is the label. It is a binary value, 1 for anomaly and -1 for normal.

    @param arr: numpy array, the dataset to be reshaped
    @param n: int, the number of rows to be aggregated
    @param sliding_window: bool, whether to use a sliding window. 
    If not using a sliding window, the rows of the output will be 1/n of the input.
    If use a sliding window, the rows of the output will be the same as the input, 
    and each row will be the aggregation of the previous n rows.
    For example, if input is [[1], [1], [1], [10], [10], [10], [1], [1], [1], [1]] and n is 2, and we are summing the column, 
    then the output will be: [[2], [11], [20], [2], [2]] without sliding_window.
    With sliding window, the output will be: [[1], [2], [2], [11], [20], [20], [11], [2], [2], [2]]
    Respectively, if the label column of the input is [[-1], [-1], [-1], [1], [1], [1], [-1], [-1], [-1], [-1]] and n is 2,
    the output will be: [[-1], [1], [1], [-1], [-1]] without sliding_window.
    With sliding window, the output will be: [[-1], [-1], [-1], [1], [1], [1], [1], [-1], [-1], [-1]]
"""
def aggregate_rows(arr, n, sliding_window=False, agg_type="origin"):
    # Apply different aggregation functions to different columns
    coloumns_to_keep = [0, 1, 2, 3]
    # Set items with feature number.
    columns_to_sum = [1, 2, 3, 4, 8, 9, 10, 12, 34, 35, 36]
    columns_to_mean = [5, 7, 13, 37]
    columns_to_max = [6, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]  
    # Make it index of columns.
    columns_to_sum = [x+3 for x in columns_to_sum]
    columns_to_mean = [x+3 for x in columns_to_mean]
    columns_to_max = [x+3 for x in columns_to_max]

    if sliding_window:
        return aggregate_rows_with_sliding_window(arr, n, coloumns_to_keep, columns_to_sum, columns_to_mean, columns_to_max, agg_type=agg_type)
    else:
        return aggregate_rows_directly(arr, n, coloumns_to_keep, columns_to_sum, columns_to_mean, columns_to_max, agg_type=agg_type )

def aggregate_rows_directly(arr, n, coloumns_to_keep, columns_to_sum, columns_to_mean, columns_to_max, agg_type="origin"):
    # Calculate the number of rows to keep to ensure the number of rows is a multiple of n
    rows_to_keep = (arr.shape[0] // n) * n
    arr = arr[:rows_to_keep]
    
    # Reshape the array to the shape (m/n, n, number of columns)
    reshaped = arr.reshape(-1, n, arr.shape[1])
    
    aggregated = np.zeros((reshaped.shape[0], reshaped.shape[2]))

    if agg_type == "median":
        aggregated[:, coloumns_to_keep] = reshaped[:, 0, coloumns_to_keep]  # Apply values of the first row in the group
        aggregated[:, columns_to_sum] = np.median(reshaped[:, :, columns_to_sum], axis=1)
        aggregated[:, columns_to_mean] = np.median(reshaped[:, :, columns_to_mean], axis=1)
        aggregated[:, columns_to_max] = np.median(reshaped[:, :, columns_to_max], axis=1)
    else: 
        aggregated[:, coloumns_to_keep] = reshaped[:, 0, coloumns_to_keep]  # Apply values of the first row in the group
        aggregated[:, columns_to_sum] = reshaped[:, :, columns_to_sum].sum(axis=1)
        aggregated[:, columns_to_mean] = reshaped[:, :, columns_to_mean].mean(axis=1)
        aggregated[:, columns_to_max] = reshaped[:, :, columns_to_max].max(axis=1)

    
    aggregated[:, -1] = np.round(np.sum(reshaped[:, :, -1] == 1, axis=1)/n)*2-1  # Apply mean and round to the label
    
    return aggregated

def aggregate_rows_with_sliding_window(arr, n, coloumns_to_keep, columns_to_sum, columns_to_mean, columns_to_max, agg_type="origin"):
    
    aggregated = np.zeros(arr.shape)    
    for i in range(arr.shape[0]):
        j = i+1
        sliding_window = arr[j-(min(j,n)):j]
        aggregated[i, coloumns_to_keep] = arr[i, coloumns_to_keep] # Just keep the value.

        if agg_type == "median":
            aggregated[:, columns_to_sum] = np.median(sliding_window[:, columns_to_sum], axis=1)
            aggregated[:, columns_to_mean] = np.median(sliding_window[:, columns_to_mean], axis=1)
            aggregated[:, columns_to_max] = np.median(sliding_window[:, columns_to_max], axis=1)
        else:  
            aggregated[i, columns_to_sum] = sliding_window[:, columns_to_sum].sum(axis=0)
            aggregated[i, columns_to_mean] = sliding_window[:, columns_to_mean].mean(axis=0)
            aggregated[i, columns_to_max] = sliding_window[:, columns_to_max].max(axis=0)
        
        aggregated[i, -1] = np.round(np.sum(sliding_window[:, -1] == 1)/n)*2-1
        
    
    return aggregated

def aggregate_datasets(datasets, time_span, sliding_window=False, agg_type="origin" ):
    aggregated_datasets = []
    for dataset in datasets:
        aggregated_datasets.append(aggregate_rows(dataset, time_span, sliding_window, agg_type=agg_type))
    return aggregated_datasets

# Test the function
if __name__ == '__main__':
    # Test the function
    print("============================================")
    arr = np.array([[1,1,1,1,-1], 
                    [1,1,1,1,-1], 
                    [1,1,1,1,-1], 
                    [10,10,10,10,1], 
                    [10,10,10,10,1], 
                    [10,10,10,10,1], 
                    [1,1,1,1,-1], 
                    [1,1,1,1,-1], 
                    [1,1,1,1,-1], 
                    [1,1,1,1,-1]])
    n = 2
    columns_to_keep = [0]
    columns_to_sum = [1]
    columns_to_mean = [2]
    columns_to_max = [3]
    print("Test the function aggregate_rows without sliding window")
    print("The input array is:")
    print(arr)
    output = aggregate_rows_directly(arr, n, columns_to_keep, columns_to_sum, columns_to_mean, columns_to_max)
    print("columns_to_keep")
    print("Expected output: 1, 1, 10, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_keep].flatten()))))
    print("columns_to_sum")
    print("Expected output: 2, 11, 20, 2, 2")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_sum].flatten()))))
    print("columns_to_mean")
    print("Expected output: 1, 5, 10, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_mean].flatten()))))
    print("columns_to_max")
    print("Expected output: 1, 10, 10, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_max].flatten()))))
    print("Label")
    print("Expected output: -1, -1, 1, -1, -1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, -1].flatten()))))
    print("--------------------------------------------")
    print("Test the function aggregate_rows with sliding window")
    output = aggregate_rows_with_sliding_window(arr, n, columns_to_keep, columns_to_sum, columns_to_mean, columns_to_max)
    print("columns_to_keep")
    print("Expected output: 1, 1, 1, 10, 10, 10, 1, 1, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_keep].flatten()))))
    print("columns_to_sum")
    print("Expected output: 1, 2, 2, 11, 20, 20, 11, 2, 2, 2")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_sum].flatten()))))
    print("columns_to_mean")
    print("Expected output: 1, 1, 1, 5, 10, 10, 5, 1, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_mean].flatten()))))
    print("columns_to_max")
    print("Expected output: 1, 1, 1, 10, 10, 10, 10, 1, 1, 1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, columns_to_max].flatten()))))
    print("Label")
    print("Expected output: -1, -1, -1, -1, 1, 1, -1, -1, -1, -1")
    print("Actual   output:", ", ".join(map(str, map(int, output[:, -1].flatten()))))
    print("--------------------------------------------")
    print("Test sliding window with n=1")
    n = 1
    output = aggregate_rows_with_sliding_window(arr, n, columns_to_keep, columns_to_sum, columns_to_mean, columns_to_max)
    print("Expected output:", arr)
    print("Actual   output:", output)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from scipy.stats import zscore

def norm( train_x, test_x, scaleType = "Std" ):
    if scaleType == "zscore" :
        print("zscore")
        train_x = zscore(train_x, axis=0, ddof=1);  # For each feature, mean = 0 and std = 1
        test_x = zscore(test_x, axis=0, ddof=1);  # For each feature, mean = 0 and std = 1
        return train_x, test_x
    
    
    if scaleType == "Std" :
        scaler = StandardScaler()
    elif scaleType == "MinMax" :
        scaler = MinMaxScaler()
    elif scaleType == "Robust" :    
        scaler = RobustScaler() 
    elif scaleType == "Power" :    
        scaler = PowerTransformer()   
    elif scaleType == "MinMax" :  
        scaler = MinMaxScaler()
    else :    
        scaler = StandardScaler()
        
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    # scaler.fit(test_x)
    if( test_x is not None ):
        test_x = scaler.transform(test_x)
    return train_x, test_x

# deal with outliers
def xpr_outlier( train_x, test_x, replacer = "upper"  ):
    outliers = []
    for i in range( train_x.shape[1] ):
        q1 = np.percentile(train_x[:, i], 25)
        q3 = np.percentile(train_x[:, i], 75)

        upper_bound = q3 + 1.5 * (q3 - q1)
        replace_value = upper_bound
        if replacer == "upper":
            replace_value = upper_bound
        elif replacer == "median":
            replace_value = np.median(train_x[:, i])
        else:
            replace_value = np.mean(train_x[:, i])
        train_x[:, i] = np.where(train_x[:, i] > upper_bound, replace_value, train_x[:, i])
        test_x[:,i] = np.where(test_x[:,i] > upper_bound, replace_value, test_x[:,i])

        return train_x, test_x

