import glob
import os
import sys
import json
from pathlib import Path

import pandas as pd


def flatten(file):
    df = pd.read_csv(file)

    def process_row(row):
        input_block = json.loads(row['input'].replace('\'', '\"'))
        option_ask = input_block['option_ask']
        option_bid = input_block['option_bid']
        ivol = input_block['volatility']
        est1, est2 = [float(i) for i in row['estimates'].strip('[ ]').split(None)]
        real1, real2 = [float(i) for i in row['real'].strip('[ ]').split(', ')]
        result = pd.DataFrame([[
            *option_ask,
            *option_bid,
            input_block['stock_ask'],
            input_block['stock_bid'],
            1,
            *ivol,
            est1,
            est2,
            real1,
            real2
        ]], columns=[
            'OPTION_ASK-2',
            'OPTION_ASK-1',
            'OPTION_ASK0',
            'OPTION_BID-2',
            'OPTION_BID-1',
            'OPTION_BID0',
            'STOCK_ASK0',
            'STOCK_BID0',
            'BIAS',
            'IVOL-2',
            'IVOL-1',
            'IVOL0',
            'EST+1',
            'EST+2',
            'REAL+1',
            'REAL+2'
        ])
        return result

    return pd.concat([process_row(row) for _, row in df.iterrows()], ignore_index=True)


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print('Usage: python predict.py [folder]')
        sys.exit(-1)
    folder = sys.argv[1]
    output_columns = {'OPTION_ASK-2': [], 'OPTION_ASK-1': [], 'OPTION_ASK0': [],
                      'OPTION_BID-2': [], 'OPTION_BID-1': [], 'OPTION_BID0': [],
                      'STOCK_ASK0': [], 'STOCK_BID0': [], 'BIAS': [],
                      'IVOL-2': [], 'IVOL-1': [], 'IVOL0': [],
                      'EST+1': [], 'EST+2': [], 'REAL+1': [], 'REAL+2': []}
    output = pd.DataFrame(output_columns)

    if os.path.isdir(folder):
        output = pd.concat([flatten(file) for file in glob.glob(f'{folder}/*.csv')], ignore_index=True)
    else:
        output = flatten(folder)
    filename = Path(folder).stem
    output_file = f'../output/flatten/{filename}_flatten.csv'
    output.to_csv(output_file, index=False)
