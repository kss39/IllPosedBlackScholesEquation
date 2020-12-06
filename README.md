# Ill Posed Black-Scholes-Merton Equation

### Run a demo
Run `python test.py`.


### NOTE
This module is mathematically sound and ready to be deployed.
The README file will be updated with instructions will be updated very soon.


## Usage

### Datablock
A datablock represents three consecutive days of option ask and bid price, the corresponding stock ask and bid price of the day, and three consecutive days of implied volatility. A datablock is declared as follows:
```
block = ns.DataBlock(today='10/19/2016',\
                       option_ask = [0.86, 0.86, 0.86],\
                       option_bid = [0.84, 0.85, 0.85],\
                       volatility = [39.456, 38.061, 37.096],\
                       stock_ask = 4.66,\
                       stock_bid = 4.65)
```

### System generation
Once three days of data are loaded in a block, the system to be solved is ready to be generated. It takes to specify the size of the matrix to fill, and the regularization parameter $\beta$:
```
m = 15
beta = 0.01
block.create_system(m, beta)
```

### System resolution
When the matrix is created and filled, the Datablock class implements its resolution:
```
result = block.solve()
```

### Solution reshape
The solution needs to be reshape for a more consistent usage:
```
solution = result.x.reshape((m-1,m-2))
print('Minimizer is:')
print(solution)

print('Estimates for 1tau and 2tau: ', solution[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)])
```

### Trading strategy
If the number returned by `solution[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)]` are greater than the ask price for tomorrow and the day after tomorrow plus \$0.03, we need to buy 5000 or 10000 options to sell them tomorrow and the day after tomorrow. More precisely:
1. if `solution[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)][0]` is greater than `block.u_a.at_day(1) + $0.03`, then we need to buy 5000 options today and sell them tomorrow;
2. if `solution[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)][1]` is greater than `block.u_a.at_day(2) + $0.03`, then we need to buy 5000 options today and sell them the day after tomorrow.
