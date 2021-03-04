import glob
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def append_errors(file):
#     df = pd.read_csv(file)
#     est = df['estimates']
#     real = df['real']
#     errors = []
#     for (e, r) in zip(est, real):
#         e = e.strip('[ ]').split()
#         r = r.strip('[ ]').split(',')
#         errors.append(np.abs(np.mean([(float(ev) - float(rv)) / float(rv) for (ev, rv) in (e, r)])))
#     return errors

def error_estimation(file):
    df = pd.read_csv(file)
    est1, est2 = df['est+1'], df['est+2']
    real1 = np.mean(df[['option_ask+1', 'option_bid+1']], axis=1)
    real2 = np.mean(df[['option_ask+2', 'option_bid+2']], axis=1)
    errors1 = []
    errors2 = []
    for (e, r) in zip(est1, real1):
        errors1.append(np.abs((e - r) / r))
    for (e, r) in zip(est2, real2):
        errors2.append(np.abs((e - r) / r))
    return errors1, errors2


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print(f'Usage: python {sys.argv[0]} [file]')
        sys.exit(-1)
    file = sys.argv[1]
    error1, error2 = error_estimation(file)
    # if os.path.isdir(folder):
    #     for file in glob.glob(f'{folder}/*.csv'):
    #         print(f'Processing {file}:')
    #         error_value_list.extend(append_errors(file))
    # else:
    #     for file in glob.glob(folder):
    #         print(f'Processing {file}:')
    #         error_value_list.extend(append_errors(file))
    print(f'length: {len(error1)}')
    print(f'median1: {np.median(error1)}')
    print(f'median2: {np.median(error2)}')
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].hist(np.array(error1), density=False, bins=100, range=(0.0, 0.8), log=False)  # `density=False` would make counts
    axs[0].set_title('t+1 prediction')
    axs[1].hist(np.array(error1), density=False, bins=100, range=(0.0, 0.8), log=False)
    axs[1].set_title('t+2 prediction')
    for ax in axs:
        ax.set(xlabel='Absolute Error', ylabel='Count')
        ax.set_xlim([0.0, 0.8])
    plt.show()
    # plt.title('Absolute Errors of 50k Estimates Compared to Real Data')
    # plt.savefig('../output/graphs/histogram_50k_log.png')
