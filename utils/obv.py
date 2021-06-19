import numpy as np

def on_balance_volume(volumes:list, close_prices:list):
    if len(volumes) != len(close_prices):
        min_len = min(len(volumes), len(close_prices))
        volumes = volumes[:min_len]
        close_prices = close_prices[:min_len]
    prev_close = 0
    obv = 0
    obv_list = []
    for volume, close_price in zip(volumes, close_prices):
        if close_price > prev_close:
            obv += volume
        elif close_price < prev_close:
            obv -= volume
        prev_close = close_price
        obv_list.append(obv)
    return np.array(obv_list)
