import math
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from model import num_solver as ns

grid = 20
beta = 0.75
folder = "./resources/tests"
output_folder = "./output/tests"


def process_csv_file(file):
    df = pd.read_csv(file).T
    output_dictionary = {'option_name': [], 'grid_count': [], 'beta': [], 'date': [], 'minimizer': []}
    output_dt = pd.DataFrame(output_dictionary)
    max_column = df.columns.stop
    for option_index in range(2, max_column, 7):
        option_name = df[option_index][0]
        day_count = len(df[option_index])
        for i in range(2, day_count - 2):
            today = df.iat[i + 2, option_index]
            data_block = df.iloc[i:i + 3, option_index + 1:option_index + 6]
            data_block = data_block.apply(pd.to_numeric)
            if data_block.isnull().values.any() is False:
                # Then the 3-day data is good for us
                option_ask = data_block.iloc[:, 1].values
                option_bid = data_block.iloc[:, 2].values
                volatility = data_block.iloc[:, 0].values
                stock_ask = math.ceil(data_block.iat[2, 3] * 100) / 100
                stock_bid = math.floor(data_block.iat[2, 4] * 100) / 100
                if stock_ask == stock_bid:
                    break
                input_data = ns.DataBlock(today=today, option_ask=option_ask, option_bid=option_bid,
                                          volatility=volatility, stock_ask=stock_ask, stock_bid=stock_bid)
                input_data.create_system(grid, beta)
                res = input_data.solve()
                row = {'option_name': option_name, 'grid_count': grid, 'beta': beta, 'date': today,
                       'minimizer': np.array2string(res.x, max_line_width=np.inf, threshold=np.inf)}
                output_dt = output_dt.append(row, ignore_index=True)
                print(f'Solved {option_name}')
            else:
                print(f'{option_name} on {today} is skipped due to insufficient data.')
    filename = Path(file).stem
    output_file = f'{output_folder}/{filename}_out.csv'
    output_dt.to_csv(output_file)


for file in glob.glob(f'{folder}/p1.csv'):
    print('Processing', file, ': ')
    process_csv_file(file)
