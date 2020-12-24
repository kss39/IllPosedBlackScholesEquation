# Ill Posed Black-Scholes-Merton Equation Solver

## Setup

Clone the `prod` branch repository: `https://github.com/kss39/IllPosedBlackScholesEquation`,
and then at the repository root folder, run `pip install .`

Now you have a Python module called `ipbse` ready to import. The `ipbse.predict` file contains the most useful function: `predict()`. To use it, just do `from ipbse import predict as ip`.

(Note: It is recommended to use environment managers like `virtualenv` or `conda`.)


## Run a demo
Run `python demo.py`. This Python script calculates a single "data block" - three consecutive trading days and predict the option price for the upcoming two days. The data here is synthetic (not real market data).

## Example of processing a folder containing csv files
`example.py` contains the code below. It will fetch all csv files in the folder and process then sequentially. There is a `test_data.csv` in the folder for test use.
```
from pathlib import Path
from glob import glob
from ipbse import predict as ip

if __name__ == '__main__':
    for file in glob('*.csv'):
        filename = Path(file)
        print(f'Processing file {file}:')
        ip.predict(file, cpu_count=4).to_csv(f'{filename.parents[0]}/{filename.stem}_prediction.csv', index=False)
```
