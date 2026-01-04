'''Black Scholes Model Operation File'''
import math
from scipy.stats import norm  # type: ignore


class EuropeanCallOption:
    '''
    EuropeanCallOption priced using the famous Black Scholes Model.
    '''
    def __init__(self, strikeprice:float, volatility:float,rate:float) -> None:
        '''
        strikeprice -> K
        rate -> r
        volatility -> sigma
        '''
        self.strikeprice = strikeprice
        self.volatility = volatility
        self.rate = rate
    def d1_term(self, spot:float, maturity:float) -> float:
        '''Black Scholes D1 term
        arguments: spot -> S0 (current price of the underlying stock)
        maturity -> T (time to maturity)'''
        if maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        d1 = (math.log(spot/self.strikeprice) + (self.rate + 0.5*self.volatility**2)*
              maturity)/(self.volatility*math.sqrt(maturity))
        return d1
    def d2_term(self, spot: float, maturity:float) -> float:
        '''Black Scholes D2 term
        arguments: spot -> S0 (current price of the underlying stock.)
        maturity -> T (time to maturity)'''
        d2 = self.d1_term(spot,maturity) - self.volatility * math.sqrt(maturity)
        return d2
    def price(self,spot:float,maturity:float) -> float:
        '''Black Scholes call option price.
        arguments: spot -> S0 (current price of the underlying stock.)
        maturity -> T (time to maturity)'''
        if maturity <=0:
            return max(spot - self.strikeprice,0.0)
        calloption_price = spot*norm.cdf(self.d1_term(spot,maturity)) - (self.strikeprice * math.e
                                                          ** (-self.rate*maturity)
                                                          ) * norm.cdf(self.d2_term(spot,maturity))
        return calloption_price
    def delta(self, spot:float,maturity:float) ->float:
        '''Option Delta
        arguments: spot -> S0 (current price of the underlying stock.)
        maturity -> T (time to maturity)
        '''
        if maturity <=0:
            return 1.0 if spot > self.strikeprice else 0.0
        option_delta = float(norm.cdf(self.d1_term(spot,maturity)))
        return option_delta
    def gamma(self, spot: float, maturity:float) -> float:
        '''Call option gamma
        arguments: spot -> S0 (current price of the underlying stock.)
        maturity -> T (time to maturity)
        '''
        if maturity <=0:
            return 0.0
        option_gamma = float(norm.pdf(self.d1_term(spot,maturity)
                                ) / (spot * self.volatility * math.sqrt(maturity)))
        return option_gamma
