import math
import sys

import pandas as pd
import numpy as np
import glob
from pathlib import Path
from model import num_solver as ns

def predict(file, grid_count, beta):
    df = pd.read_csv(file)
    output_dictionary = {'option_name': [], 'input': [], 'grid_count': [], 'beta': [], 'date': [], 'estimates': [], 'real': []}
    output_dt = pd.DataFrame(output_dictionary)
    day_count = len(df)
    for i in range(2, day_count-2):
        today = df['DATE'][i]
        data_block = df.iloc[i - 2:i + 1]
        if len(set(data_block['OPTION_NAME'])) != 1:
            continue
        else:
            option_name = data_block['OPTION_NAME'].iat[0]
        value_block = data_block[['EOD_OPTION_PRICE_ASK',
                                  'EOD_OPTION_PRICE_BID',
                                  'IVOL_LAST',
                                  'EOD_UNDERLYING_PRICE_ASK',
                                  'EOD_UNDERLYING_PRICE_BID']]
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
            solution = res.x.reshape((m - 1, m - 2))[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)]
            real_future = []
            for j in range(2):
                future_price = df['EOD_OPTION_PRICE_LAST'][i+j]
                if np.isnan(future_price):
                    future_price = np.mean(df[['EOD_OPTION_PRICE_ASK', 'EOD_OPTION_PRICE_BID']].iloc[i+j, :])
                real_future.append(future_price)
            row = {'option_name': option_name, 'input': input_data.data(), 'grid_count': grid_count, 'beta': beta, 'date': today,
                   'estimates': solution, 'real': real_future}
            output_dt = output_dt.append(row, ignore_index=True)
            print(f'Solved {option_name} on {today}, finished {i-1}/{day_count-4}')
        else:
            print(f'{option_name} on {today} is skipped due to insufficient data; finished {i-1}/{day_count-4}')
    filename = Path(file).stem
    output_file = f'output/prediction/{filename}_prediction.csv'
    output_dt.to_csv(output_file)


if __name__ == '__main__':
    if not len(sys.argv) == 4:
        print('Usage: python predict.py [grid_count] [beta] [folder]')
        sys.exit(-1)
    grid_count = int(sys.argv[1])
    beta = float(sys.argv[2])
    folder = sys.argv[3]
    for file in glob.glob(f'{folder}/*.csv'):
        print(f'Processing {file}:')
        predict(file, grid_count, beta)
