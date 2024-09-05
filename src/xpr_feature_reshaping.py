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
"""
def aggregate_rows(arr, n):
    # Calculate the number of rows to keep to ensure the number of rows is a multiple of n
    rows_to_keep = (arr.shape[0] // n) * n
    arr = arr[:rows_to_keep]
    
    # Reshape the array to the shape (m/n, n, number of columns)
    reshaped = arr.reshape(-1, n, arr.shape[1])
    
    # Apply different aggregation functions to different columns
    coloumns_to_keep_the_first = [0, 1, 2, 3]
    # Set items with feature number.
    columns_to_sum = [1, 2, 3, 4, 8, 9, 10, 12, 34, 35, 36]
    columns_to_mean = [5, 7, 13, 37]
    columns_to_max = [6, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]  
    # Make it index of columns.
    columns_to_sum = [x+3 for x in columns_to_sum]
    columns_to_mean = [x+3 for x in columns_to_mean]
    columns_to_max = [x+3 for x in columns_to_max]
    
    aggregated = np.zeros((reshaped.shape[0], reshaped.shape[2]))
    aggregated[:, coloumns_to_keep_the_first] = reshaped[:, 0, coloumns_to_keep_the_first]  # Apply values of the first row in the group
    aggregated[:, columns_to_sum] = reshaped[:, :, columns_to_sum].sum(axis=1)
    aggregated[:, columns_to_mean] = reshaped[:, :, columns_to_mean].mean(axis=1)
    aggregated[:, columns_to_max] = reshaped[:, :, columns_to_max].max(axis=1)
    aggregated[:, 41] = np.round(np.sum(reshaped[:, :, 41] == 1, axis=1)/n)*2-1  # Apply mean and round to the label
    
    return aggregated

def aggregate_datasets(datasets, time_span):
    aggregated_datasets = []
    for dataset in datasets:
        aggregated_datasets.append(aggregate_rows(dataset, time_span))
    return aggregated_datasets
