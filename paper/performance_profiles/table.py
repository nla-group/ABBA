################################################################################
# Generate csv from pickle file.

filename = 'scl0.p'

################################################################################

import os
import pickle
import pandas as pd

D = pickle.load(open(filename, 'rb'))
measures = ['2', 'DTW', '2_diff', 'DTW_diff']
del D['k'], D['scl'], D['tol']

# Save to csv
df = pd.DataFrame.from_dict(D,orient='index').transpose()
df.to_csv('full_scl0.csv', index=False)

# Remove failed tests
partial_df = df.loc[df['error'] == 0]
temp = partial_df['ts_name'].str.split('.tsv_')
temp = pd.DataFrame(temp.tolist(), index=temp.index)
temp = temp.drop_duplicates([0])

partial_df = partial_df[partial_df.index.isin(temp.index)]
partial_df.to_csv('partial_scl0.csv', index=False)
