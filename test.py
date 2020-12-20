from model import num_solver as ns

block = ns.DataBlock(today='10/19/2016',\
                       option_ask = [0.86, 0.86, 0.86],\
                       option_bid = [0.84, 0.85, 0.85],\
                       volatility = [39.456, 38.061, 37.096],\
                       stock_ask = 4.66,\
                       stock_bid = 4.65)

m = 15
beta = 0.01

block.create_system(m, beta)
result = block.solve()
solution = result.x.reshape((m-1,m-2))
print('Minimizer is:')
print(solution)

print('Estimates for 1tau and 2tau: ', solution[[math.ceil(m/2-1), m-2], math.ceil((m-2)/2)])
