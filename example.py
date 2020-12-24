from pathlib import Path
from glob import glob
from ipbse import predict as ip

if __name__ == '__main__':
    for file in glob('*.csv'):
        filename = Path(file)
        print(f'Processing file {file}:')
        ip.predict(file, cpu_count=4).to_csv(f'{filename.parents[0]}/{filename.stem}_prediction.csv', index=False)
