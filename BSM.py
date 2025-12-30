import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import datetime as dt
sns.set_theme(style="whitegrid", palette="colorblind")
import warnings
warnings.filterwarnings('ignore')

class BSM:
    def __init__(self, asset):
        self.asset = asset

    def black_scholes_price(self, TYPE="CALL", S=100, K=100, r=0.05, t=0.4, s=0.01) -> float:
        d1 = (np.log(S / K) + t * (r + (s ** 2) / 2)) / (s * np.sqrt(t))
        d2 = (np.log(S / K) + t * (r - (s ** 2) / 2)) / (s * np.sqrt(t))

        price = 0

        if TYPE == "CALL":
            price = S * scipy.stats.norm.cdf(x=d1) - K * np.exp(-r * t) * scipy.stats.norm.cdf(x=d2)
        elif TYPE == "PUT":
            price = K * np.exp(-r * t) * scipy.stats.norm.cdf(x=-d2) - S * scipy.stats.norm.cdf(x=-d1)

        return price

    def stock_option_evolution(self, TYPE = "CALL", K = 250, t = 60) -> None:
        STOCK = yf.download(tickers=self.asset, period="10y", auto_adjust=True)["Close"]
        STOCK["LOG CHANGE"] = np.log(STOCK / STOCK.shift(1))
        STOCK.columns = ["CLOSE", "LOG CHANGE"]

        STOCK.dropna(axis=0, inplace=True)
        log_mean = STOCK["LOG CHANGE"].mean()
        log_std = STOCK["LOG CHANGE"].std(ddof=1)
        annual_std = log_std * np.sqrt(252)

        DTE = t
        STRIKE = K

        # IMPORT CURRENT YIELD ON 10 YEAR T-BOND
        R = yf.download(tickers="^TNX", period="1d", auto_adjust=True).Close.iloc[-1].iloc[0] / 100

        time = [0]
        start_price = STOCK["CLOSE"].iloc[-1]
        stock_prices = [start_price]
        call_prices = [self.black_scholes_price(TYPE, S = start_price, K = K, r = R, t = (DTE + 1) / 365, s = annual_std)]
        P_L = 0

        for i in range(1, DTE + 1):
            price_change = scipy.stats.t.rvs(df=5, loc=log_mean, scale=log_std)
            next_stock_price = stock_prices[-1] * np.exp(price_change)
            stock_prices.append(next_stock_price)

            T = (DTE - i + 1) / 365

            call_price = self.black_scholes_price(TYPE, S = next_stock_price, K = K, r = R, t = T, s = annual_std)
            call_prices.append(call_price)
            time.append(i)

            if i == DTE - 1:
                P_L = next_stock_price - STRIKE

        print(f" -> P/L OF HOLDING {TYPE} OPTION: ${P_L:.3f}")
        results = pd.DataFrame(index=time, data={"STOCK": stock_prices, "CALL": call_prices})
        sns.lineplot(data=results, dashes=False)
        plt.hlines(y=STRIKE, colors='m', xmin=0, xmax=DTE, label="STRIKE")
        plt.vlines(x=DTE, ymin=STRIKE if (stock_prices[-1] > STRIKE) else stock_prices[-1],
                   ymax=STRIKE if (stock_prices[-1] < STRIKE) else stock_prices[-1],
                   colors='g' if (stock_prices[-1] > STRIKE) else 'r', label="P/L")
        plt.xlabel("DAYS")
        plt.ylabel("PRICE ($)")
        plt.title("STOCK-CALL PRICE EVOLUTION")
        plt.legend()
        plt.show()
        return None

    def black_scholes_individual(self) -> None:
        # STOCK DATA
        STOCK = yf.download(tickers = self.asset, period="10y", auto_adjust=True)["Close"]
        STOCK["LOG CHANGE"] = np.log(STOCK / STOCK.shift(1))
        STOCK.columns = ["CLOSE", "LOG CHANGE"]
        STOCK.dropna(axis=0, inplace=True)

        log_mean = STOCK["LOG CHANGE"].mean()
        log_std = STOCK["LOG CHANGE"].std(ddof=1)

        # IMPORT CURRENT YIELD ON 10 YEAR T-BOND FOR RISK-FREE RATE
        R = yf.download(tickers="^TNX", period="1d", auto_adjust=True).Close.iloc[-1].iloc[0] / 100

        # OPTIONS DATA
        TICKER = yf.Ticker(ticker=self.asset)
        TICKER._download_options()
        expirations = list(TICKER._expirations.keys())

        if len(expirations) == 0:
            print("OPTIONS DATA UNAVAILABLE...")
            return None

        soonest_date = expirations[-1]
        calls = TICKER.option_chain(date=soonest_date).calls
        liquid_calls = calls.loc[calls['volume'] > 0]

        if len(liquid_calls) == 0:
            print("NO LIQUID OPTIONS AVAILABLE...")
            return None

        expiry_date = dt.date(int(soonest_date[0:4]), int(soonest_date[5:7]), int(soonest_date[8:]))
        VOL = log_std * np.sqrt(252)
        DTE = (expiry_date - dt.date.today()).days
        STRIKE = liquid_calls["strike"].iloc[0]
        S_0 = STOCK["CLOSE"].iloc[-1]

        QUOTE_PRICE = (liquid_calls["bid"].iloc[0] + liquid_calls["ask"].iloc[0]) / 2
        BSM_PRICE = self.black_scholes_price(TYPE = "CALL", S = S_0, K = STRIKE, r = R, t = DTE, s = VOL)

        print(f"QUOTE = ${QUOTE_PRICE:.3f}")
        print(f"BSM = ${BSM_PRICE:.3f}")

        if BSM_PRICE > QUOTE_PRICE:
            print("BSM says option is ...undervalued...")
        else:
            print("BSM says option is ...overvalues...")

        return None

    def black_scholes_calls(self) -> None:
        # STOCK DATA
        STOCK = yf.download(tickers=self.asset, period="10y", auto_adjust=True)["Close"]
        STOCK["LOG CHANGE"] = np.log(STOCK / STOCK.shift(1))
        STOCK.columns = ["CLOSE", "LOG CHANGE"]
        STOCK.dropna(axis=0, inplace=True)

        log_std = STOCK["LOG CHANGE"].std(ddof=1)

        # IMPORT CURRENT YIELD ON 10 YEAR T-BOND FOR RISK-FREE RATE
        R = yf.download(tickers="^TNX", period="1d", auto_adjust=True).Close.iloc[-1].iloc[0] / 100

        # OPTIONS DATA
        TICKER = yf.Ticker(ticker=self.asset)
        TICKER._download_options()
        expirations = list(TICKER._expirations.keys())

        if len(expirations) == 0:
            print("OPTIONS DATA UNAVAILABLE...")
            return None

        VOL = log_std * np.sqrt(252)
        S_0 = STOCK["CLOSE"].iloc[-1]

        UNDER = 0
        OVER = 0
        CALL_COUNT = 0

        for i in range(len(expirations)):
            expiration_date = expirations[i]
            calls = TICKER.option_chain(date=expiration_date).calls
            calls = calls.loc[calls['volume'] > 0]

            if len(calls) == 0:
                continue

            EXP = dt.date(int(expiration_date[0:4]), int(expiration_date[5:7]), int(expiration_date[8:]))
            DTE = (EXP - dt.date.today()).days / 365

            for j in range(calls.shape[0]):
                CALL_COUNT += 1

                STRIKE = calls["strike"].iloc[j]
                QUOTE_PRICE = (calls["bid"].iloc[j] + calls["ask"].iloc[j]) / 2
                BSM_PRICE = self.black_scholes_price("CALL", S_0, STRIKE, R, DTE, VOL)

                if (BSM_PRICE < QUOTE_PRICE):
                    UNDER += 1
                else:
                    OVER += 1

        result = pd.DataFrame({"UNDERVALUED": [UNDER], "OVERVALUED": [OVER]})
        sns.barplot(data=result)
        plt.title(f"BSM OVERVIEW OF {CALL_COUNT} {self.asset} CALL OPTIONS")
        plt.ylabel("COUNTS")
        plt.show()
        return None

    def black_scholes_puts(self) -> None:
        # STOCK DATA
        STOCK = yf.download(tickers=self.asset, period="10y", auto_adjust=True)["Close"]
        STOCK["LOG CHANGE"] = np.log(STOCK / STOCK.shift(1))
        STOCK.columns = ["CLOSE", "LOG CHANGE"]
        STOCK.dropna(axis=0, inplace=True)

        log_std = STOCK["LOG CHANGE"].std(ddof=1)

        # IMPORT CURRENT YIELD ON 10 YEAR T-BOND FOR RISK-FREE RATE
        R = yf.download(tickers="^TNX", period="1d", auto_adjust=True).Close.iloc[-1].iloc[0] / 100

        # OPTIONS DATA
        TICKER = yf.Ticker(ticker=self.asset)
        TICKER._download_options()
        expirations = list(TICKER._expirations.keys())

        if len(expirations) == 0:
            print("OPTIONS DATA UNAVAILABLE...")
            return None

        VOL = log_std * np.sqrt(252)
        S_0 = STOCK["CLOSE"].iloc[-1]

        UNDER = 0
        OVER = 0
        CALL_COUNT = 0

        for i in range(len(expirations)):
            expiration_date = expirations[i]
            puts = TICKER.option_chain(date=expiration_date).puts
            puts = puts.loc[puts['volume'] > 0]

            if len(puts) == 0:
                continue

            EXP = dt.date(int(expiration_date[0:4]), int(expiration_date[5:7]), int(expiration_date[8:]))
            DTE = (EXP - dt.date.today()).days / 365

            for j in range(puts.shape[0]):
                CALL_COUNT += 1

                STRIKE = puts["strike"].iloc[j]
                QUOTE_PRICE = (puts["bid"].iloc[j] + puts["ask"].iloc[j]) / 2
                BSM_PRICE = self.black_scholes_price("PUT", S_0, STRIKE, R, DTE, VOL)

                if BSM_PRICE < QUOTE_PRICE:
                    UNDER += 1
                else:
                    OVER += 1

        result = pd.DataFrame({"UNDERVALUES": [UNDER], "OVERVALUED": [OVER]})
        sns.barplot(data=result)
        plt.title(f"BSM OVERVIEW OF {CALL_COUNT} {self.asset} PUT OPTIONS")
        plt.ylabel("COUNTS")
        plt.show()
        return None

    def implied_vol_surface(self) -> None:
        TICKER = yf.Ticker(ticker=self.asset)
        TICKER._download_options()
        expirations = list(TICKER._expirations.keys())

        if len(expirations) == 0:
            print("OPTIONS DATA UNAVAILABLE...")
            # return None

        DTEs = []
        STRIKEs = []
        IVs = []

        for i in range(len(expirations)):
            EXP = dt.date(int(expirations[i][0:4]), int(expirations[i][5:7]), int(expirations[i][8:]))
            expiration_date = expirations[i]

            calls = TICKER.option_chain(date=expiration_date).calls
            calls = calls.loc[calls['volume'] > 0]

            puts = TICKER.option_chain(date=expiration_date).puts
            puts = puts.loc[puts['volume'] > 0]

            if calls.shape[0] != 0:
                for j in range(calls.shape[0]):
                    STRIKE = calls["strike"].iloc[j]
                    IV = calls["impliedVolatility"].iloc[j]

                    STRIKEs.append(STRIKE)
                    DTEs.append((EXP - dt.date.today()).days)
                    IVs.append(IV)

            if puts.shape[0] != 0:
                for j in range(puts.shape[0]):
                    STRIKE = puts["strike"].iloc[j]
                    IV = puts["impliedVolatility"].iloc[j]

                    STRIKEs.append(STRIKE)
                    DTEs.append((EXP - dt.date.today()).days)
                    IVs.append(IV)

        df = pd.DataFrame(data={'Strike': STRIKEs, 'DTE': DTEs, 'IV': IVs})
        df = df[(df['IV'] > 0) & (df['IV'] < 5)]

        xi = np.linspace(df['Strike'].min(), df['Strike'].max(), 60)
        yi = np.linspace(df['DTE'].min(), df['DTE'].max(), 60)

        X, Y = np.meshgrid(xi, yi)

        # Interpolate unstructured data onto the grid
        Z = griddata((df['Strike'], df['DTE']), df['IV'], (X, Y), method='linear')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               linewidth=0, antialiased=True)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.set_xlabel('Strike', fontsize=11)
        ax.set_ylabel('Days to Expiration', fontsize=11)
        ax.set_zlabel('Implied Volatility', fontsize=11)
        ax.set_title(f'{self.asset} Implied Volatility Surface from {df.shape[0]} OPTIONS', fontsize=13)

        fig.colorbar(surf)
        plt.show()
        return None