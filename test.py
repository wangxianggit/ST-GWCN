import h5py
import numpy as np
import pandas as pd

filename = 'metr-la.h5'
f = h5py.File(filename, 'r')

# List all groups
# print("Keys: %s" % f.keys())
# a_group_key = list(f.keys())[0]
#
# # Get the data
# data = list(f[a_group_key])
list = [f]
data = pd.DataFrame([list])
data.to_csv("hi.csv")