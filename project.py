import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries import offsets
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
from numpy import fft
#plt.style.use('')
#[u'seaborn-darkgrid', u'seaborn-notebook', u'classic', u'seaborn-ticks', u'grayscale', u'bmh', u'seaborn-talk', u'dark_background', u'ggplot', 
#u'fivethirtyeight', u'seaborn-colorblind', u'seaborn-deep', u'seaborn-whitegrid', 
#u'seaborn-bright', u'seaborn-poster', u'seaborn-muted', u'seaborn-paper', u'seaborn-white', u'seaborn-pastel', u'seaborn-dark', u'seaborn-dark-palette']

N_SAMPLE = 100
N_AHEAD = 4
if_oil = False
if if_oil:
    oil = pd.read_csv("data/DCOILWTICO.csv")
    n_start = 10
    n_end = len(oil)
    N_DATA = n_end - n_start
    oil_values = np.array(oil["Value"][n_start:n_end])
    futures = pd.read_csv("data/NYMEX.csv", sep='\s+',header=None)
    futures1 = np.array(futures[3])[n_start:n_end]
    futures2 = np.array(futures[4])[n_start:n_end]
    futures3 = np.array(futures[5])[n_start:n_end]
    futures4 = np.array(futures[6])[n_start:n_end]
    futures = [futures1, futures2, futures3, futures4]
    data = oil_values
    start_date = oil["Date"][n_start + N_SAMPLE - 1]
    title = "WTI Crude Oil"
else:
    SnP500 = pd.read_csv("data/SnP500.csv")
    futures = None
    n_start =  0
    print("very first date =", SnP500["Date"][len(SnP500) - 1 - n_start])
    n_end = len(SnP500)
    N_DATA = n_end - n_start
    SnP500_values = np.array(SnP500["Adj Close"][::-1][n_start:n_end])
    data = SnP500_values
    start_date = SnP500["Date"][len(SnP500) - 1 - (n_start + N_SAMPLE - 1)]
    title = "S&P500"

print("start_date =", start_date)
N_PRED = N_DATA - N_SAMPLE - N_AHEAD + 1
print("N =", N_DATA)
print("N_AHEAD =", N_AHEAD)
print("sample size =", N_SAMPLE)
print("# of predictions made", N_PRED)
# N_SAMPLE -- length of samples that are used to make predictions
# N_AHEAD -- the number of future months' prices predicted
# N_PRED = N_DATA - N_SAMPLE - N_AHEAD + 1 --  the number of predictions made
# wavelet decomposition parameters
N_levels = 5
wavelet = pywt.Wavelet('db7') #['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']
mode = "sym"  #['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per']
# strategy constants
MAX_DRAWDOWN = 1.0
FRICTION = 0.01


def decomposition(data, n_levels, wavelet, mode):
    """
    Wavelet decomposition into n_levels levels
    data = a[n_levels - 1] + d[0] + ... + d[n_levels - 1]
    """
    Ca = []
    Cd = []
    Ca_i = np.copy(data)
    for i in range(n_levels):
        (Ca_i, Cd_i) = pywt.dwt(Ca_i, wavelet=wavelet, mode=mode)
        Ca.append(Ca_i)
        Cd.append(Cd_i)
    a = []
    d = []
    for i in range(n_levels):
        coeff_a = [Ca[i], None] + [None] * i
        coeff_d = [None, Cd[i]] + [None] * i
        a.append(pywt.waverec(coeff_a, wavelet=wavelet, mode=mode)[:len(data)])
        d.append(pywt.waverec(coeff_d, wavelet=wavelet, mode=mode)[:len(data)])
    data_rec = np.copy(a[-1])
    for i in range(n_levels):
        data_rec += d[i]
    return a, d, data_rec


def plot_decomposition(data, a, d, data_rec, title):
    """
    Plots wavelet decomposition of the signal
    """
    fig = plt.figure()
    ax_main = fig.add_subplot(len(a) + 1, 2, 1)
    ax_main.set_title(title)
    ax_main.set_ylabel("signal")
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)
    for i in range(N_levels):
        ax = fig.add_subplot(len(a) + 1, 2, 3 + i * 2)
        ax.plot(a[i], 'r')
        ax.set_xlim(0, len(a[i]) - 1)
        ax.set_ylabel("a%d" % (i + 1))
    for i in range(N_levels):
        ax = fig.add_subplot(len(d) + 1, 2, 4 + i * 2)
        ax.plot(d[i], 'g')
        ax.set_xlim(0, len(d[i]) - 1)
        ax.set_ylabel("d%d" % (i + 1))
    ax_main = fig.add_subplot(len(a) + 1, 2, 2)
    ax_main.plot(data_rec)
    ax_main.set_ylabel("rec")
    ax_main.set_xlim(0, len(data_rec) - 1)


def sine_fit(t, p1, p2, p3, p4):
    return p1 * np.sin(p2 * t + p3) + p4

def sine_trending_fit(t, p1, p2, p3, p4, p5):
    return p1 * np.sin(p2 * t + p3) + p4 + p5 * t

def fourierExtrapolation(x, n_predict, sort='freq'):
    """
    from https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
    """
    n = x.size
    n_harm = 10                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    if sort == 'freq':
        # sort indexes by frequency, lower -> higher
        indexes.sort(key = lambda i: np.absolute(f[i]))
    elif sort == 'ampl':
        # sort indexes by amplitudes, higher -> lower
        indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
        indexes.reverse()
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


def wavelet_prediction(a, d, N_SAMPLE, N_AHEAD, a_spline_order=5, d_spline_order=5, start=0, method="Fourier"):
    """
    Extrapolates a[n_levels - 1] and d[n_levels - 1] using spline fit and d[i] using sine fit or Fourier fit into N_AHEAD steps in the future and then
    resonctructs the prediction for the original signal
    """
    #print len(a[0]), N_SAMPLE
    time = np.arange(start, N_SAMPLE + start)
    time_future = np.arange(N_SAMPLE + start, N_SAMPLE + N_AHEAD + start)
    a_extrapolation = UnivariateSpline(time, a[-1], k=a_spline_order)(time_future)
    data_rec_prediction = np.copy(a_extrapolation)
    data_rec_prediction += UnivariateSpline(time, d[-1], k=d_spline_order)(time_future)
    for i in range(N_levels - 1):
        if method == "Fourier":
            d_extrapolation = fourierExtrapolation(d[i], N_AHEAD)[-1 - N_AHEAD:-1]
        elif method == "sine": 
            fitting_parameters, covariance = curve_fit(sine_fit, time, d[i], maxfev=3000)
            d_extrapolation = sine_fit(time_future, *fitting_parameters)
        else:
            raise ValueError("extrapolation method must be either Fourier or sine")
        data_rec_prediction += d_extrapolation
    return data_rec_prediction


def get_data(data):
    """
    data -- original array of prices of length N_DATA
    data_today -- array of length N_PRED that contains actual spot prices for the days when predictions were made
    data_ahead -- array N_AHEAD x N_PRED, data_ahead[j][:] contains (j + 1) months ahead prices
    """
    data_today = data[N_SAMPLE - 1: N_SAMPLE - 1 + N_PRED]
    data_ahead = np.empty(shape=(N_AHEAD, N_PRED))
    for j in range(N_AHEAD):
        data_ahead[j][:] = data[N_SAMPLE + j: N_SAMPLE + j + N_PRED]
    return data_today, data_ahead


def get_prediction(data, n_levels, wavelet, mode, a_spline_order=5, d_spline_order=5, method="Fourier"):
    """
    data -- original array of prices of length N_DATA
    data_prediction -- array N_AHEAD x N_PRED, data_prediction[j][:] contains wavelet based prediction for (j + 1) months ahead prices
    """
    data_prediction = np.empty(shape=(N_AHEAD, N_PRED))
    for i in range(N_PRED):
        for j in range(N_AHEAD):
            a, d, data_rec = decomposition(data[i: i + N_SAMPLE], n_levels=n_levels, wavelet=wavelet, mode=mode)
            data_prediction[j][i] = wavelet_prediction(a, d, N_SAMPLE=N_SAMPLE, N_AHEAD=N_AHEAD, a_spline_order=a_spline_order, d_spline_order=d_spline_order, start=0, method=method)[j]
    return data_prediction


def get_returns(data_today, data_ahead, data_prediction):
    """
    returns_data -- array N_AHEAD x N_PRED - 1, returns_data[j][:] contains (j + 1) months ahead actual returns
    returns_prediction -- array N_AHEAD x N_PRED - 1, returns_prediction[j][:] contains (j + 1) months ahead predicted returns
    """
    returns_data = np.zeros(shape=(N_AHEAD, N_PRED - 1))
    returns_prediction = np.zeros(shape=(N_AHEAD, N_PRED - 1))
    for j in range(N_AHEAD):
        returns_data[j][:] = data_ahead[j][:-1] / data_today[:-1] - 1.0
        returns_prediction[j][:] = data_prediction[j][:-1] / data_today[:-1] - 1.0
    return returns_data, returns_prediction


def get_trailing_vols(data):
    """
    returns trailing volatility of the actual returns based on N_SAMPLE points before the day predictions are made
    """
    trailing_vols = []
    for i in range(N_PRED):
        data_i = data[i:i + N_SAMPLE]
        trailing_vols.append(np.std(data_i[1:] / data_i[:-1] - 1.0))
    return np.array(trailing_vols)


def get_balance(data, returns_data, returns_prediction):
    """
    returns account balance of initial $1 invested and traded using our strategy and passive investing into the index
    balance_data -- array of length N_PRED, contains balance of initial $1 invested in passive index
    balance_strategy -- array N_AHEAD x N_PRED - 1, balance_strategy[j][:] contains strategy balance based on (j + 1) months ahead wavelet prediction
    """
    balance_data = np.ones(N_PRED)
    balance_strategy = np.ones(shape=(N_AHEAD, N_PRED))
    returns_strategy = np.zeros(shape=(N_AHEAD, N_PRED - 1))
    vols = get_trailing_vols(data=data)
    for i in range(1, N_PRED):
        balance_data[i] = balance_data[i - 1] * (1.0 + returns_data[0][i - 1])
        for j in range(N_AHEAD):
            if returns_data[j][i - 1] < 0:
                r = (0.01 * returns_data[j][i - 1] + 0.99 * returns_prediction[j][i - 1])
            elif returns_data[j][i - 1] > 0:
                r = (0.2 * returns_data[j][i - 1] + 0.8 * returns_prediction[j][i - 1])
            if returns_prediction[j][i - 1] == 0.0:
                r = 0.0
            weight = MAX_DRAWDOWN * (1.0 - 2.0 * norm.cdf(-r / vols[i - 1])) * (np.abs(returns_prediction[j][i - 1]) < 0.07)
            #weight *= (weight > 0)
            #weight = np.sign(returns_prediction[j][i - 1])
            #weight *= (weight > 0)#(np.abs(returns_prediction[j][i - 1]) < 0.1)
            #weight = -np.sign(returns_prediction[j][i - 1])
            balance_strategy[j][i] = (1 - weight) * balance_strategy[j][i - 1] +  (1.0 + returns_data[j][i - 1]) * balance_strategy[j][i - 1] * weight - FRICTION * weight * balance_strategy[j][i - 1]
            returns_strategy[j][i - 1] = balance_strategy[j][i] / balance_strategy[j][i - 1] - 1.0
    return balance_data, balance_strategy, returns_strategy


def annual_Sharpe(returns, freq=12):
    """
    Returns annualized Sharpe ratio
    """
    return np.sqrt(freq) * np.mean(returns) / np.std(returns)


def max_drawdown(data):
    """
    Computes max drawdown
    http://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
    """
    mdd = 0
    peak = data[0]
    for x in data:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd  

def plot_prediction(data_actual, data_prediction, title, print_only=False):
    """
    Plots predicted data vs actual and prints R values
    """
    print(title)
    for j in range(N_AHEAD):
        x = np.array(data_actual[j])
        y = np.array(data_prediction[j])
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        if not print_only:
            fig = plt.figure()
            plt.plot(x, y, 'o')
            plt.title("%d months " % (j + 1) + title + " vs actual" +  ", r_value = %f" % r_value)
            plt.grid(True)
            p2, = plt.plot(x, intercept + slope * x)
            plt.legend([p2], ["r_value = %f" % r_value])
        print (j + 1), "months:", "r_value =", r_value



def plot_returns(returns_strategy, returns_prediction, print_only=False):
    """
    Plots actual and expected returns and prints some statistics
    """
    print("trading summary")
    for i in range(len(returns_strategy)):
        if not print_only:
            if (i == 1) or (i == 3):
                plt.figure()
                plt.title("%d months predicted returns" % (i + 1))
                p1, = plt.plot(returns_strategy[i])
                p2, = plt.plot(returns_prediction[i])
                plt.legend([p1, p2], ["actual returns", "predicted returns"])
                plt.title("%d months predicted returns" % (i + 1))
        total_days = N_PRED
        win_days = sum(e > 0 for e in returns_strategy[i])
        lose_days = sum(e < 0 for e in returns_strategy[i])
        rest_days = sum(e == 0 for e in returns_strategy[i])
        win_rate = float(win_days) / (win_days + lose_days)
        rest_rate = float(rest_days) / total_days 
        trade_days = win_days + lose_days
        correlation = np.corrcoef(returns_strategy[i], returns_prediction[i])[0][1]
        return_avr = np.mean(returns_strategy[i])
        return_vol = np.std(returns_strategy[i])
        print (i + 1), "months:", "total =", total_days, "days_trade =", trade_days, "win =", win_days, "lose =", lose_days, "rest =", rest_days, \
        "percentage of winnigs =", "{0:.2f}".format(win_rate), "rest_rate =", "{0:.3f}".format(rest_rate), \
        "return_avr =", "{0:.4f}".format(return_avr), "return_vol =", "{0:.4f}".format(return_vol), "correlation =", "{0:.3f}".format(correlation)


def plot_balance(balance_data, balance_strategy, print_only=False):
    """
    Plots account balance of initial $1 invested and traded using our strategy
    """
    print("account balance")
    for j in range(N_AHEAD):
        time = pd.date_range(start=pd.Timestamp(start_date) + offsets.DateOffset(months=j), periods=N_PRED, freq='M')
        max_drawdown_data = 100 * max_drawdown(balance_data)
        max_drawdown_strategy = 100 * max_drawdown(balance_strategy[j][:])
        if not print_only:
            if (j == 2) or (j == 4):
                plt.figure()
                p1, = plt.plot(time, balance_data)
                p2, = plt.plot(time, balance_strategy[j])
                plt.title("balance for %d months strategy" % (4))
                plt.legend([p1, p2], ["sharpe_data", "sharpe_strategy"])
        print (j + 1), "months:", "return =", "{0:.3f}".format(100 * (balance_strategy[j][-1] - 1.0)), "return_data =", "{0:.3f}".format(100 * (balance_data[-1] - 1.0)), \
              "Strategy Max drawdown =", "{0:.3f}".format(max_drawdown_strategy), "Max drawdown S&P500 =", "{0:.3f}".format(max_drawdown_data)


def plot_sharpe(returns_data, returns_strategy, print_only=False):
    """
    Plots Sharpe ratio for passive investing and our strategy
    """
    print("Sharpe ratio")
    sharpe_data = [annual_Sharpe(returns_data[0][:i]) for i in range(1, len(returns_data[0][:]))]
    for j in range(N_AHEAD):
        time = pd.date_range(start=pd.Timestamp(start_date) + offsets.DateOffset(months=(j + 1)), periods=N_PRED, freq='M')
        sharpe_strategy = [annual_Sharpe(returns_strategy[j][:i]) for i in range(1, len(returns_strategy[j][:]))]
        #drawdown_strategy = 100 * np.abs(min(returns_strategy[j][:]))
        #drawdown_data = 100 * np.abs(min(returns_data[0][:]))
        if not print_only:
            if (j == 1) or (j == 3):
                plt.figure()
                plt.title("%d months Sharpe ratio" % (j + 1))
                p1, = plt.plot(sharpe_data)
                p2, = plt.plot(sharpe_strategy)
                plt.ylim([-2.0, 2.0])
                plt.legend([p1, p2], ["sharpe_data", "sharpe_strategy"])
        print (j + 1), "months:", "Sharpe_strategy =", "{0:.3f}".format(sharpe_strategy[-1]), "Sharpe_data =", "{0:.3f}".format(sharpe_data[-1])
        


a, d, data_rec = decomposition(data=data, n_levels=N_levels, wavelet=wavelet, mode=mode)
plot_decomposition(data, a, d, data_rec, title=title) 
data_today, data_ahead = get_data(data=data)
data_prediction_1 = get_prediction(data=data, n_levels=N_levels, wavelet=wavelet, mode=mode, a_spline_order=5, d_spline_order=5, method="Fourier")
data_prediction_2 = get_prediction(data=data, n_levels=N_levels, wavelet=wavelet, mode=mode, a_spline_order=1, d_spline_order=5, method="Fourier")
data_prediction_3 = get_prediction(data=data, n_levels=N_levels, wavelet=wavelet, mode=mode, a_spline_order=2, d_spline_order=5, method="Fourier")
returns_data, returns_prediction_1 = get_returns(data_today=data_today, data_ahead=data_ahead, data_prediction=data_prediction_1)
returns_data, returns_prediction_2 = get_returns(data_today=data_today, data_ahead=data_ahead, data_prediction=data_prediction_2)
returns_data, returns_prediction_3 = get_returns(data_today=data_today, data_ahead=data_ahead, data_prediction=data_prediction_3)
returns_prediction = np.where((np.sign(returns_prediction_1) == np.sign(returns_prediction_2)) & ((np.sign(returns_prediction_2) == np.sign(returns_prediction_3))), 0.333 * (returns_prediction_1 + returns_prediction_2 + returns_prediction_3), 0.0)
returns_prediction = np.where(np.sign(returns_prediction_1) != np.sign(returns_prediction_2), 0.0, 0.5 * (returns_prediction_1 + returns_prediction_2))
balance_data, balance_strategy, returns_strategy = get_balance(data=data, returns_data=returns_data, returns_prediction=returns_prediction)
plot_balance(balance_data, balance_strategy, print_only=False)
plot_sharpe(returns_data, returns_strategy, print_only=True)
plot_returns(returns_strategy, returns_prediction, print_only=True)

#plot_prediction(returns_data, returns_prediction, title="wavelet prediction", print_only=True)

#plot_prediction(data_ahead, data_prediction, title="wavelet prediction", print_only=False)
#print returns_data
if if_oil:
    data_prediction = [futures[i][N_SAMPLE - 1:N - N_AHEAD - 1] for i in range(N_AHEAD)] 
    plot_prediction(data_future_actual=data_ahead, data_prediction=data_prediction, title="futures")


plt.show()


