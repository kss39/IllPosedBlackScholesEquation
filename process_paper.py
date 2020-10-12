import math
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from model import num_solver as ns

grid = 20
beta = 2
folder = "./resources/paper_data"
output_folder = "./output/paper_data"


def process_csv_file(file):
    df = pd.read_csv(file)
    output_dictionary = {'option_name': [], 'input': [], 'grid_count': [], 'beta': [], 'date': [], 'minimizer': []}
    output_dt = pd.DataFrame(output_dictionary)
    day_count = len(df)
    for i in range(2, day_count - 2):
        today = df.iat[i + 2, 0]
        data_block = df.iloc[i:i + 3]
        # data_block = data_block.apply(pd.to_numeric)
        option_name = 'unknown'
        if not data_block.isnull().values.any():
            # Then the 3-day data is good for us
            option_ask = data_block.iloc[:, 1].values
            option_bid = data_block.iloc[:, 2].values
            volatility = data_block.iloc[:, 4].values
            stock_ask = math.ceil(data_block.iat[2, 5] * 100) / 100
            stock_bid = math.floor(data_block.iat[2, 6] * 100) / 100
            if stock_ask == stock_bid:
                stock_ask += 0.01
            input_data = ns.DataBlock(today=today, option_ask=option_ask, option_bid=option_bid,
                                      volatility=volatility, stock_ask=stock_ask, stock_bid=stock_bid)
            input_data.create_system(grid, beta)
            res = input_data.solve()
            row = {'option_name': option_name, 'input': str(input_data), 'grid_count': grid, 'beta': beta, 'date': today,
                   'minimizer': np.array2string(res.x, max_line_width=np.inf, threshold=np.inf)}
            output_dt = output_dt.append(row, ignore_index=True)
            print(f'Solved {option_name} on {today}')
        else:
            print(f'{option_name} on {today} is skipped due to insufficient data.')
    filename = Path(file).stem
    output_file = f'{output_folder}/{filename}_out.csv'
    output_dt.to_csv(output_file)


for file in glob.glob(f'{folder}/p1.csv'):
    print('Processing', file, ': ')
    process_csv_file(file)
