import numpy as np

class Smoother():
    def __init__(self, data:list, window_size:int=7, alpha:float=None):
        self.data = data
        self.window_size = window_size
        self.alpha = alpha
        self.ma_types = ['exponential', 'hamming', 'harmonic', 'simple']

    def __capture_window(self, data, window_size):
        i = 0
        windows = []
        while i < len(data) - window_size+1:
            window = data[i:i+window_size]
            windows.append(window)
            i += 1
        return windows

    def transform(self, ma_type='hamming'):
        if not ma_type.lower() in self.ma_types:
            raise Exception(f'Unsupport moving average type.\nSupport types[{self.ma_types}]')
        data = self.data
        n = self.window_size
        moving_average = []
        if ma_type == 'exponential':
            if type(self.alpha) == type(None):
                alpha = 2/(n+1) # Smoothing Factor (SF)
            else:
                alpha = self.alpha
            for idx in range(len(data)):
                if idx == 0:
                    moving_average.append(data[idx])
                    continue
                else:
                    ema = alpha*data[idx]+(1-alpha)*moving_average[idx-1]
                    moving_average.append(ema)
        else:
            windows_data = self.__capture_window(data, n)
            for window in windows_data:
                if ma_type == 'harmonic':
                    # Harmonic Moving Average
                    divider = 0
                    for x in window:
                        divider += 1/x
                    hma = n/divider
                    moving_average.append(hma)
                elif ma_type == 'hamming':
                    # Need to fix
                    # Hamming Moving Average
                    weights = np.arange(1, n+1)
                    hma = np.dot(window, weights)/weights.sum()
                    moving_average.append(hma)
                elif ma_type =='simple':
                    # Need to fix
                    # Simple Moving Average
                    moving_average.append(window.sum()/n)
        return np.array(moving_average)
