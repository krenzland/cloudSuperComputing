#!/usr/bin/env python3
import glob
import pandas as pd
from collections import defaultdict

def main():
    files = list(sorted(glob.glob('*/solution.csv')))
    print(files)
    # dicts are order : value
    data = defaultdict(list)
    results = []
    for file in files:
        df = pd.read_csv(file)
        df = df[df['norm'] == 'l2']
        #print(df.columns)
        order = df['order'].values[0]
        hmin = df['hmin'].values[0]
        error = df['var3'].values[0]
        #print(order, error)
        data[order].append((hmin, error))
        results.append([order, hmin, error])

    df = pd.DataFrame(results)
    df.columns = ['order', 'hmin', 'error']
    print(df)
    df.to_csv('errors.csv', index=None)

    for order, values in data.items():
        print("\n{}".format(order))
        error_sum = 0.0
        error_count = 0
        for hmin, error in values:
            print("{} {}".format(hmin, error))
            error_sum += error
            error_count +=1
        error_mean = error_sum/error_count
        print()
        print(error_mean)

if __name__ == '__main__':
    main()
