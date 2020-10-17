import math
import sys
import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import glob
from pathlib import Path
from model import num_solver as ns


output_dictionary = {'option_name': [], 'input': [], 'grid_count': [], 'beta': [], 'date': [], 'estimates': [],
                     'real': []}
value_list = ['EOD_OPTION_PRICE_ASK',
              'EOD_OPTION_PRICE_BID',
              'IVOL_LAST',
              'EOD_UNDERLYING_PRICE_ASK',
              'EOD_UNDERLYING_PRICE_BID']

grid_count = 20
beta = 0.01


def predict(file):
    df = pd.read_csv(file)

    day_count = len(df)
    # # TODO: Debug!
    # day_count = 20
    manager = mp.Manager()
    output_dt_lock = manager.Lock()
    namespace = manager.Namespace()
    namespace.df = pd.DataFrame(output_dictionary)
    with mp.Pool(processes=mp.cpu_count(), initargs=(output_dt_lock,)) as pool:
        pool.starmap(solve, [(i, df, namespace, output_dt_lock, day_count) for i in range(2, day_count-2)])
    filename = Path(file).stem
    output_file = f'output/prediction/{filename}_prediction.csv'
    namespace.df.to_csv(output_file)


def solve(i, df, namespace, output_lock, day_count):
    today = df['DATE'][i]
    data_block = df.iloc[i - 2:i + 1]
    if len(set(data_block['OPTION_NAME'])) != 1:
        return
    else:
        option_name = data_block['OPTION_NAME'].iat[0]
    value_block = data_block[value_list]
    if not value_block.isnull().values.any():
        # Then the 3-day data is good for us
        option_ask = value_block['EOD_OPTION_PRICE_ASK'].values
        option_bid = value_block['EOD_OPTION_PRICE_BID'].values
        volatility = value_block['IVOL_LAST'].values
        stock_ask = float(value_block['EOD_UNDERLYING_PRICE_ASK'].iat[2])
        stock_bid = float(value_block['EOD_UNDERLYING_PRICE_BID'].iat[2])
        input_data = ns.DataBlock(today=today, option_ask=option_ask, option_bid=option_bid,
                                  volatility=volatility, stock_ask=stock_ask, stock_bid=stock_bid)
        input_data.create_system(grid_count, beta)
        res = input_data.solve()
        m = grid_count
        solution = res.x.reshape((m - 1, m - 2))[[math.ceil(m / 2 - 1), m - 2], math.ceil((m - 2) / 2)]
        real_future = []
        for j in range(2):
            future_price = df['EOD_OPTION_PRICE_LAST'][i + j]
            if np.isnan(future_price):
                future_price = np.mean(df[['EOD_OPTION_PRICE_ASK', 'EOD_OPTION_PRICE_BID']].iloc[i + j, :])
            real_future.append(future_price)
        row = {'option_name': option_name, 'input': input_data.data(), 'grid_count': grid_count, 'beta': beta,
               'date': today,
               'estimates': solution, 'real': real_future}
        output_lock.acquire()
        try:
            namespace.df = namespace.df.append(row, ignore_index=True)
            print(f'Solved {option_name} on {today}, finished {i - 1}/{day_count - 4}')
        finally:
            output_lock.release()
    else:
        print(f'{option_name} on {today} is skipped due to insufficient data; finished {i - 1}/{day_count - 4}')


# Time testing: grid=20, beta=0.01, 16 blocks, 64.26s
# After disabling Numpy parallel: 28.36s
if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print('Usage: python predict.py [grid_count] [beta] [folder]')
        sys.exit(-1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    grid_count = int(sys.argv[1])
    beta = float(sys.argv[2])
    folder = sys.argv[3]
    for file in glob.glob(f'{folder}/*.csv'):
        print(f'Processing {file}:')
        predict(file)
