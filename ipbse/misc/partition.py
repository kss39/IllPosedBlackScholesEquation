import glob
import pandas as pd


def partition(df, chunk_size=3000):
    for index, start in enumerate(range(0, len(df), chunk_size)):
        df_subset = df.iloc[start:start+chunk_size]
        output_file = f'../resources/partitioned/partitioned_{index}.csv'
        df_subset.to_csv(output_file, index=False)


if __name__ == '__main__':
    while True:
        directory = input('Please enter the folder, or Ctrl-C to stop')
        total_frame = pd.concat([pd.read_csv(file) for file in glob.glob(f'{directory}/*.csv')])
        partition(total_frame)
