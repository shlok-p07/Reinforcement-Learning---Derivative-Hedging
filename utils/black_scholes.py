'''Black Scholes Model Operation File'''
import math
from scipy.stats import norm  # type: ignore


class EuropeanCallOption:
    '''
    EuropeanCallOption priced using the famous Black Scholes Model.
    '''
    def __init__(self, strikeprice:float, maturity:float,volatility:float,rate:float) -> None:
        '''
        strikeprice -> K
        maturity -> T
        rate -> r
        volatility -> sigma
        '''
        self.strikeprice = strikeprice
        self.maturity = maturity
        self.volatility = volatility
        self.rate = rate
    def d1_term(self, spot:float) -> float:
        '''Black Scholes D1 term
        argument: spot -> S0 (current price of the underlying stock)'''
        d1 = (math.log(spot/self.strikeprice) + (self.rate + 0.5*self.volatility**2)*
              self.maturity)/(self.volatility*math.sqrt(self.maturity))
        return d1
    def d2_term(self, spot: float) -> float:
        '''Black Scholes D2 term
        argument: spot -> S0 (current price of the underlying stock.)'''
        d2 = self.d1_term(spot) - self.volatility * math.sqrt(self.maturity)
        return d2
    def price(self,spot:float) -> float:
        '''Black Scholes call option price.
        argument: spot -> S0 (current price of the underlying stock.)'''
        calloption_price = spot*norm.cdf(self.d1_term(spot)) - (self.strikeprice * math.e
                                                          ** (-self.rate*self.maturity)
                                                          ) * norm.cdf(self.d2_term(spot))
        return calloption_price
    def delta(self, spot:float) ->float:
        '''Option Delta'''
        option_delta = float(norm.cdf(self.d1_term(spot)))
        return option_delta
