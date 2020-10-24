import math
import glob
import pandas as pd
from pathlib import Path


def preprocess(filename):
    data = pd.read_csv(filename).T
    output_dictionary = {'OPTION_NAME': [],
                         'DATE': [],
                         'EOD_OPTION_PRICE_LAST': [],
                         'EOD_OPTION_PRICE_ASK': [],
                         'EOD_OPTION_PRICE_BID': [],
                         'IVOL_LAST': [],
                         'EOD_UNDERLYING_PRICE_ASK': [],
                         'EOD_UNDERLYING_PRICE_BID': []}
    df = pd.DataFrame(output_dictionary)
    max_column = data.columns.stop
    for option_index in range(2, max_column, 7):
        option_name = data[option_index][0]
        day_count = len(data[option_index])
        for i in range(2, day_count):
            today = data.iat[i, option_index]
            data_block = data.iloc[i, option_index + 1:option_index + 6]
            data_block = data_block.apply(pd.to_numeric)
            if not data_block.isnull().values.any():
                option_ask = data_block.iat[1]
                option_bid = data_block.iat[2]
                volatility = data_block.iat[0] * 100
                stock_ask = math.ceil(data_block.iat[3] * 100) / 100
                stock_bid = math.floor(data_block.iat[4] * 100) / 100
                if stock_ask == stock_bid:
                    stock_ask += 0.01
                row = {'OPTION_NAME': option_name,
                       'DATE': today,
                       'EOD_OPTION_PRICE_LAST': None,
                       'EOD_OPTION_PRICE_ASK': option_ask,
                       'EOD_OPTION_PRICE_BID': option_bid,
                       'IVOL_LAST': volatility,
                       'EOD_UNDERLYING_PRICE_ASK': stock_ask,
                       'EOD_UNDERLYING_PRICE_BID': stock_bid}
                df = df.append(row, ignore_index=True)
                print(f'Preprocessed {option_name} at day {today}')
            else:
                print(f'{option_name} on {today} is skipped due to insufficient data.')
    filename = Path(filename).stem
    output_file = f'../resources/preprocessed/{filename}_preprocessed.csv'
    df.to_csv(output_file)


if __name__ == '__main__':
    while True:
        directory = input('Please enter the filename, or Ctrl-C to stop')
        for file in glob.glob(f'{directory}/*.csv'):
            print(f'Processing {file}:')
            preprocess(file)
