import glob
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def append_errors(file):
    df = pd.read_csv(file)
    est = df['estimates']
    real = df['real']
    errors = []
    for (e, r) in zip(est, real):
        e = e.strip('[ ]').split()
        r = r.strip('[ ]').split(',')
        errors.append(np.abs(np.mean([(float(ev) - float(rv)) / float(rv) for (ev, rv) in (e, r)])))
    return errors


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print(f'Usage: python {sys.argv[0]} [folder]')
        sys.exit(-1)
    error_value_list = []
    folder = sys.argv[1]
    if os.path.isdir(folder):
        for file in glob.glob(f'{folder}/*.csv'):
            print(f'Processing {file}:')
            error_value_list.extend(append_errors(file))
    else:
        for file in glob.glob(folder):
            print(f'Processing {file}:')
            error_value_list.extend(append_errors(file))
    plt.hist(error_value_list, density=False, bins='auto')  # `density=False` would make counts
    plt.ylabel('Count')
    plt.xlabel('Absolute Error')
    plt.xlim(xmin=0.0, xmax=0.8)
    plt.show()

