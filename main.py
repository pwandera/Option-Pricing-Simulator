from BSM import *

def print_menu():
    print("\n|------------------------------------------------------MENU--------------------------------------------------------------------|")

    # BSM.price_simulator()
    print('1. GENERATE OPTION PRICE')
    print(' -> Input an option type, strike price, quote price, and time to expiry to calculate fair value given current stock price.\n')

    # BSM.stock_option_evolution
    print('2. GENERATE STOCK-OPTION EVOLUTION')
    print(' -> Input an option type, strike price, and time to expiry to see how it evolves with the stock until expiry.\n')

    # BSM.black_scholes_calls() and BSM.black_scholes_puts()
    print('3. VISUALISE WHAT BSM MODEL THINKS ABOUT ALL CURRENT STOCK OPTIONS ON THE MARKET')
    print(' -> Input an option type and generate a bar chart to see counts of overvalued vs. undervalued options according to BSM model.\n')

    # BSM.implied_vol_surface()
    print('4. GENERATE IMPLIED VOLATILITY SURFACE')
    print(' -> Generate an implied volatility surface from all call and put options data for the stock on the market.\n')

    print('5. EXIT PROGRAM')
    print('|--------------------------------------------------------------------------------------------------------------------------------|\n')

if __name__ == '__main__':
    print('WELCOME TO THE BLACK-SCHOLES-MERTON OPTION PRICING SIMULATOR')
    ticker = input('ENTER STOCK/INDEX TICKER TO ANALYZE: ')
    asset = BSM(ticker)
    print_menu()

    # MAIN PROGRAM LOOP
    while True:

        try:
            choice = int(input('ENTER CHOICE: '))
        except:
            print('INVALID CHOICE... TRY AGAIN.')
            continue

        if choice == 1:
            TYPES = ['CALL', 'PUT']

            try:
                TYPE = input('OPTION TYPE (CALL OR PUT): ')

                if TYPE not in TYPES:
                    print('INVALID OPTION TYPE')
                    continue

                STRIKE = float(input('STRIKE (K): $'))
                QUOTE = float(input('QUOTE: $'))
                TIME_TO_EXPIRY = float(input('TIME TO EXPIRY (T): '))
                asset.price_simulator(TYPE, QUOTE, STRIKE, TIME_TO_EXPIRY)

            except:
                print('INVALID NUMERICAL INPUT...\n')

        elif choice == 2:
            TYPES = ['CALL', 'PUT']

            try:
                TYPE = input('OPTION TYPE (CALL OR PUT): ')

                if TYPE not in TYPES:
                    print('INVALID TYPE... TRY AGAIN')

                STRIKE = int(input('STRIKE (K): $'))
                TIME_TO_EXPIRY = int(input('TIME TO EXPIRY (T): '))
                asset.stock_option_evolution(TYPE, STRIKE, TIME_TO_EXPIRY)

            except:
                print('INVALID INPUT...\n')

        elif choice == 3:
            TYPES = ['CALL', 'PUT']

            TYPE = input('OPTION TYPE (CALL OR PUT): ')

            if TYPE not in TYPES:
                print('INVALID TYPE... TRY AGAIN\n')

            if TYPE == 'CALL':
                asset.black_scholes_calls()
            else:
                asset.black_scholes_puts()

        elif choice == 4:
            asset.implied_vol_surface()

        elif choice == 5:
            print('GOODBYE!')
            break