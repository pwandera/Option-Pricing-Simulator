# Option Pricing Simulator

This project provides a Python-based framework for analyzing equity options using the ***Black-Scholes-Merton*** model. It leverages live market data from Yahoo Finance to calculate theoretical option prices, visualize implied volatility surfaces, and simulate the profit/loss evolution of holding an option over time.

## Features
***Real-Time Valuation***: Fetches live stock and option chain data to calculate theoretical BSM prices.  
***Market Comparison***: Scans entire option chains to classify options as "undervalued" or "overvalued" compared to current market asks/bids.  
***Stochastic Simulation***: Simulates the life of an option contract using t-distribution price shocks to visualize price paths and P/L evolution.  
***3D Visualization***: Generates Implied Volatility (IV) surfaces by interpolating unstructured strike/expiration data.  
***Risk-Free Rate Integration***: Automatically fetches the current 10-Year U.S. Treasury Yield (^TNX) to use as the risk-free interest rate ($r$).  

## Model  
$$ C(S, t) = S Φ(d_1) - K e^{-rt} Φ(d_2), $$
$$ P(S, t) = Ke^{-rt}Φ(-d_2) - S Φ(-d_1) $$  
  
$$ d_1 = \frac{\ln(S/K) + t(r + \frac{\sigma^2}{2})}{\sigma\sqrt{t}} $$  
  
$$AND$$  

$$ d_2 = \frac{\ln(S/K) + t(r - \frac{\sigma^2}{2})}{\sigma\sqrt{t}} = d_1 - \sigma\sqrt{t} $$  

##### Where:  
$S$ = Current Stock Price  
$K$ = Strike Price  
$r$ = Risk-free interest rate (sourced from 10-Year T-Note)  
$t$ = Time to maturity (in years)  
$\sigma$ = Volatility (derived from annualized standard deviation of log returns)  
$Φ(•)$ = Cumulative Distribution Function of a Standard Normal Random Variable  

## Assumptions & Drawbacks

1. Stock prices follow a lognormal distribution:

$$ \ln(S_T) \\sim \ N(S_0 + T(μ - \frac{\sigma^2}{2}), \sigma^2 T) $$  

This implies a constant mean and variance. However, we can only estimate these parameters using historical data and these estimates change each time we measure them. We cannot capture their true valaues. 
This fact is visualised when we observe Implied Volatility surfaces. Under the Black Scholes Model, these surfaces should be flat since volatility is assumed to be constant. 
The IV surfaces tell a different story about what the market actually thinks of volatility into the future.  

2. The derivative is a European option which cannot be exercised before the expiration date. Other mathematical frameworks are needed to price American options such as Binomial Trees.

2. The short selling of securities with full use of proceeds is permitted.

3. There are no transaction costs or taxes. All securities are perfectly divisible.

4. There are no dividends during the life of the derivative.

5. There are no riskless arbitrage opportunities.

6. Security trading is continuous.

7. The risk-free rate of interest, $r$, is constant and the same for all maturities.
