import math
import sys
import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import glob
import model.num_solver as ns

output_dictionary = {
    'option_name': [],
    'grid_count': [],
    'beta': [],
    'date': [],
    'option_ask-2': [],
    'option_ask-1': [],
    'option_ask0': [],
    'option_bid-2': [],
    'option_bid-1': [],
    'option_bid0': [],
    'stock_ask0': [],
    'stock_bid0': [],
    'ivol-2': [],
    'ivol-1': [],
    'ivol-0': [],
    'est+1': [],
    'est+2': [],
    'real+1': [],
    'real+2': []
}
# output_dictionary = {'option_name': [], 'input': [], 'grid_count': [], 'beta': [], 'date': [], 'estimates': [],
#                      'real': []}
value_list = ['EOD_OPTION_PRICE_ASK',
              'EOD_OPTION_PRICE_BID',
              'IVOL_LAST',
              'EOD_UNDERLYING_PRICE_ASK',
              'EOD_UNDERLYING_PRICE_BID']


def predict(file: str, cpu_count=1, grid_count=20, beta=0.01):
    """
    Predict the option+1 and option+2 price using the data given.
    The results are in a output csv file.

    Note that the results are in somewhat random order in time, due
    to parallelization.
    :param file: the .csv file of input
    :param cpu_count: thread count. Used for parallelization.
    :param grid_count: the grid count for each dimension.
    :param beta: beta parameter for Tikhonov regularization
    :return a Dataframe of predictions
    """
    # Temporarily disable Numpy's multithreading since
    # it will slow down multiprocessing module.
    temp_openblas = os.environ['OPENBLAS_NUM_THREADS']
    temp_mkl = os.environ['MKL_NUM_THREADS']
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    df = pd.read_csv(file)

    day_count = len(df)
    manager = mp.Manager()
    output_dt_lock = manager.Lock()
    namespace = manager.Namespace()
    namespace.df = pd.DataFrame(output_dictionary)
    with mp.Pool(processes=cpu_count, initargs=(output_dt_lock,)) as pool:
        pool.starmap(solve, [(i, df, namespace, output_dt_lock, day_count, grid_count, beta) for i in range(2, day_count-2)])

    os.environ['OPENBLAS_NUM_THREADS'] = temp_openblas
    os.environ['MKL_NUM_THREADS'] = temp_mkl
    return namespace.df


def solve(i, df, namespace, output_lock, day_count, grid_count, beta):
    if 'DATE' in df:
        today = df['DATE'][i]
    else:
        today = None
    data_block = df.iloc[i - 2:i + 1]
    if 'OPTION_NAME' in df:
        if len(set(data_block['OPTION_NAME'])) > 1:
            return
        else:
            option_name = data_block['OPTION_NAME'].iat[0]
    else:
        option_name = None
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
            if 'EOD_OPTION_PRICE_LAST' in df:
                future_price = df['EOD_OPTION_PRICE_LAST'][i + j]
            else:
                future_price = np.mean(df[['EOD_OPTION_PRICE_ASK', 'EOD_OPTION_PRICE_BID']].iloc[i + j, :])
            real_future.append(future_price)
        row = {
            'option_name': option_name,
            'grid_count': grid_count,
            'beta': beta,
            'date': today,
            'option_ask-2': input_data.u_a_list[0],
            'option_ask-1': input_data.u_a_list[1],
            'option_ask0': input_data.u_a_list[2],
            'option_bid-2': input_data.u_b_list[0],
            'option_bid-1': input_data.u_b_list[1],
            'option_bid0': input_data.u_b_list[2],
            'stock_ask0': input_data.s_a,
            'stock_bid0': input_data.s_b,
            'ivol-2': input_data.ivol_list[0],
            'ivol-1': input_data.ivol_list[1],
            'ivol-0': input_data.ivol_list[2],
            'est+1': float(solution[0]),
            'est+2': float(solution[1]),
            'real+1': float(real_future[0]),
            'real+2': float(real_future[1])
        }
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
    if not len(sys.argv) == 5:
        print('Usage: python predict.py [grid_count] [beta] [folder] [cpu_count]')
        sys.exit(-1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    grid_count = int(sys.argv[1])
    beta = float(sys.argv[2])
    folder = sys.argv[3]
    cpu_count = int(sys.argv[4])
    if os.path.isdir(folder):
        for file in glob.glob(f'{folder}/*.csv'):
            print(f'Processing {file}:')
            predict(file, cpu_count)
    else:
        predict(folder, cpu_count)

