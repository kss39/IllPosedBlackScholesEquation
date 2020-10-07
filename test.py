from model import num_solver as ns
import math


block = ns.DataBlock(today='10/19/2016',\
                       option_ask = [0.86, 0.86, 0.86],\
                       option_bid = [0.84, 0.85, 0.85],\
                       volatility = [39.456, 38.061, 37.096],\
                       stock_ask = 4.66,\
                       stock_bid = 4.65)

m = 10
print(block.create_system(m, 0.75))
result = block.solve()
solution = result.x.reshape((m-1,m-2))
print(solution)

print(solution[m-2][math.floor(m-2/2):math.ceil(m-2/2) + 1])
