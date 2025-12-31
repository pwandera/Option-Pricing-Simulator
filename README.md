# Option-Pricing-Simulator

This project provides a Python-based framework for analyzing equity options using the Black-Scholes-Merton model. It leverages live market data from Yahoo Finance to calculate theoretical option prices, visualize implied volatility surfaces, and simulate the profit/loss evolution of holding an option over time.

## Features
**Real-Time Valuation**: Fetches live stock and option chain data to calculate theoretical BSM prices.  
**Market Comparison**: Scans entire option chains to classify options as "undervalued" or "overvalued" compared to current market asks/bids.  
**Stochastic Simulation**: Simulates the life of an option contract using t-distribution price shocks to visualize price paths and P/L evolution.  
**3D Visualization**: Generates Implied Volatility (IV) surfaces by interpolating unstructured strike/expiration data.  
**Risk-Free Rate Integration**: Automatically fetches the current 10-Year U.S. Treasury Yield (^TNX) to use as the risk-free interest rate ($r$).  

## Model
$$ C(S, t) = S N(d_1) - K e^{-rt} N(d_2)$$$$P(S, t) = K e^{-rt} N(-d_2) - S N(-d_1) $$  
  
$$ d_1 = \frac{\ln(S/K) + t(r + (\sigma^2/2))}{\sigma\sqrt{t}} $$  
  
$$AND$$  

$$ d_2 = \frac{\ln(S/K) + t(r - (\sigma^2/2))}{\sigma\sqrt{t}} = d_1 - \sigma\sqrt{t} $$  

Where:  
$S$ = Current Stock Price  
$K$ = Strike Price  
$r$ = Risk-free interest rate (sourced from 10-Year T-Note)  
$t$ = Time to maturity (in years)  
$\sigma$ = Volatility (derived from annualized standard deviation of log returns)  
