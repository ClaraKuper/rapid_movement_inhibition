import numpy as np
import pandas as pd
import src.helper_funcs as helper

loc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
weights = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
array = np.array([loc, weights]).T

data = pd.DataFrame(array, columns=['location', 'weight'])

weighted_average = helper.get_weighted_average(data, 'location', 'weight')
print(weighted_average)
if weighted_average == 5:
    print('test passed')
else:
    raise ValueError('There was an issue with the weighted average')
