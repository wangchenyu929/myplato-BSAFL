import pandas as pd
import numpy as np


coll_class = np.zeros((2,3,2,2))
read_path = 'test.csv'
temp_original_data = pd.read_csv(read_path)
original_X = np.array(temp_original_data.iloc[list(range(0,200)),[2,5]]).reshape(20,20)
print(original_X)
# temp_original_X = temp_original_data.iloc([1,2],[2,5]).reshape(2,2)
# temp_original_y = temp_original_data.iloc([1,2],[3,6]).reshape(2,2)
# temp_original_z = temp_original_data.iloc([1,2],[4,7]).reshape(2,2)
# coll_class[0][0]=temp_original_X
# coll_class[0][1]=temp_original_y
# coll_class[0][2]=temp_original_z

# temp_original_X = temp_original_data.iloc([3,4],[2,5]).reshape(2,2)
# temp_original_y = temp_original_data.iloc([3,4],[3,6]).reshape(2,2)
# temp_original_z = temp_original_data.iloc([3,4],[4,7]).reshape(2,2)
# coll_class[1][0]=temp_original_X
# coll_class[1][1]=temp_original_y
# coll_class[1][2]=temp_original_z

# coll_class