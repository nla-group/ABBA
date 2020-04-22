import os
import pickle
import pandas as pd

for file in ['scl0.p', 'scl1.p']:
    fname = file[:-2]

    D = pickle.load(open(file, 'rb'))
    measures = ['2', 'DTW', '2_diff', 'DTW_diff']
    del D['k'], D['scl'], D['tol']

    # Save to csv
    df = pd.DataFrame.from_dict(D,orient='index').transpose()
    df.to_csv('full_'+fname+'.csv', index=False)

    # Remove failed tests
    partial_df = df.loc[df['error'] == 0]
    temp = partial_df['ts_name'].str.split('.tsv_')
    temp = pd.DataFrame(temp.tolist(), index=temp.index)
    temp = temp.drop_duplicates([0])

    partial_df = partial_df[partial_df.index.isin(temp.index)]
    partial_df.to_csv('partial_'+fname+'.csv', index=False)
