'''Black Scholes Operations Validation'''
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.black_scholes import EuropeanCallOption

STRIKEPRICE = 100
MATURITY = 0.5
RATE = 0.01
VOLATILITY = 0.2

option = EuropeanCallOption(STRIKEPRICE,VOLATILITY,RATE)
spots = list(range(50,151,5))
times = [t/100 for t in range(1,101)]

dvs = [option.delta(S,MATURITY) for S in spots]
gvs = [option.gamma(S,MATURITY) for S in spots]
pvs = [option.price(S,MATURITY)for S in spots]

fig,axs = plt.subplots(2,2,figsize=(12,10))

axs[0,0].plot(spots,pvs,marker='o',color='purple')
axs[0,0].set_title("Price Vs. Spot")
axs[0,0].set_xlabel("Spot Price (S)")
axs[0,0].set_ylabel("Call Price")
axs[0,0].grid(True)


axs[0,1].plot(spots,dvs,marker='o',color='blue')
axs[0,1].set_title("Delta Vs. Spot")
axs[0,1].set_xlabel("Spot Price (S)")
axs[0,1].set_ylabel("Delta")
axs[0,1].grid(True)


axs[1,0].plot(spots,gvs,marker='x',color='red')
axs[1,0].set_title("Price Vs. Spot")
axs[1,0].set_xlabel("Spot Price (S)")
axs[1,0].set_ylabel("Gamma")
axs[1,0].grid(True)

ps = {
    "Deep ITM {S=150}": option.price(150,MATURITY),
    "ATM (S=K=100)": option.price(100,MATURITY),
    "Deep OTM (S=50)": option.price(50,MATURITY)
}
axs[1,1].bar(ps.keys(),ps.values(),color=['orange','red','blue'])
axs[1,1].set_title('Price Sanity Check')
axs[1,1].set_ylabel("Option Price")
axs[1,1].grid(True, axis='y')
plt.tight_layout(rect=(0,0,1,0.96))
plt.show()
