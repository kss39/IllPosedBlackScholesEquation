# Ill Posed Black-Scholes-Merton Equation Solver

### Setup

Clone the `prod` branch repository: `https://github.com/kss39/IllPosedBlackScholesEquation/tree/prod`,
and then at the repository root folder, run `pip install .`

Now you have a Python module called `ipbse` ready to import. The `ipbse.predict_parallel` file contains the most useful function: `predict()`. To use it, just do `import ipbse.predict_parallel`.

(Note: It is recommended to use environment managers like `virtualenv` or `conda`.)


### Run a demo
Run `python demo.py`. This Python script calculates a single "data block" - three consecutive trading days and predict the option price for the upcoming two days. The data here is synthetic (not real market data).
