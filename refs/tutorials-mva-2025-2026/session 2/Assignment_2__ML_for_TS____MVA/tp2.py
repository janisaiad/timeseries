# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# **Machine Learning for Time Series (Master MVA)**
#
# - [Link to the class material.](http://www.laurentoudre.fr/ast.html)

# %% [markdown]
# # Introduction
#
# In this tutorial, we will explore several techniques for **signal modeling and denoising**.
# In particular, we will cover the following topics:
#
# - modeling a signal into trend, seasonality, and stationary components,
# - denoising a signal using standard filtering techniques,
# - predicting signals through parametric models,
# - autoregressive (AR) and moving average (MA) processes,a
# - singular spectrum analysis (SSA).

# %% [markdown]
# ## Setup
#
# **Imports**

# %%
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

import IPython.display as ipd
from scipy.signal import medfilt
from numpy.fft import rfft, rfftfreq
from numpy.polynomial.polynomial import Polynomial
from scipy.cluster import hierarchy
from scipy.signal import butter, filtfilt
from scipy.signal import argrelmax, periodogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

# %% [markdown]
# **Utility functions**

# %%
try:
    from numpy.lib.stride_tricks import sliding_window_view  # New in version 1.20.0

    def get_trajectory_matrix(arr, window_shape, jump=1):
        return sliding_window_view(x=arr, window_shape=window_shape)[::jump]

except ImportError:

    def get_trajectory_matrix(arr, window_shape, jump=1):
        n_rows = ((arr.size - window_shape) // jump) + 1
        n = arr.strides[0]
        return np.lib.stride_tricks.as_strided(
            arr, shape=(n_rows, window_shape), strides=(jump * n, n)
        )
    
def fig_ax(figsize=(15, 4)):
    return plt.subplots(figsize=figsize)

def get_largest_local_max(
    signal1D: np.ndarray, n_largest: int = 3, order: int = 1
) -> [np.ndarray, np.ndarray]:
    """Return the largest local max and the associated index in a tuple.

    This function uses `order` points on each side to use for the comparison.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]
    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_indexes = all_local_max_indexes[all_local_max.argsort()[::-1]][
        :n_largest
    ]

    return (
        np.take(signal1D, largest_local_max_indexes),
        largest_local_max_indexes,
    )

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def average_anti_diag(traj_matrix: np.ndarray) -> np.ndarray:
    """Average anti diagonal elements of a 2d array"""
    x1d = [
        np.mean(traj_matrix[::-1, :].diagonal(i))
        for i in range(-traj_matrix.shape[0] + 1, traj_matrix.shape[1])
    ]
    return np.array(x1d)


# %% [markdown]
# # Signal representation

# %% [markdown]
# In the first section of this tutorial, we will see how signal modeling can be used to create compact and efficient representations of time series data.
#
# We will begin with a brief review of the concepts of aliasing and spectral leakage.

# %% [markdown]
# ### Aliasing

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Generate four pure sine waves with the following frequencies: 261.63 Hz, 43838.37 Hz, 196 Hz, and 43904 Hz. 
#     Set the duration to 3 seconds and the sampling frequency to 44.1 kHz.</p>
#     <p>Listen to the signal. What do you observe? Explain your observation.</p>
#     <p>Then, repeat the same experiment using a sampling frequency of 88.2 kHz.</p>
#     <p>How does increasing the sampling rate affect the signal you hear ?</p>
# </div>
#

# %%
T = 3
fs= 44100
t= np.linspace(0, T, int(T * fs), endpoint=False) 
frequencies=[261.63, 43838.37, 196, 43904]
signals=[np.sin(2 * np.pi * f * t) for f in frequencies]

# %%
for i,f in enumerate(frequencies):
    ipd.display(ipd.Audio(signals[i], rate=44100))
    print(f"Played frequency: {f} Hz")

# %% [markdown]
# ### Spectral leakage

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Compute and display the DFT of a pure sine wave of frequency f=1.5 Hz.
#     Set the duration to 3 seconds and the sampling frequency to 5 Hz.</p>
#     </p>What is the expected DFT shape? What do you observe? How to cope with this phenomenon ?</p>
# </div>

# %%
fs = 5
T = 3
t = np.linspace(0, T, int(T * fs), endpoint=False)
f = 1.5
plt.stem(np.abs(np.fft.fft(np.sin(2 * np.pi * f * t))))
plt.show()

# %% [markdown]
# ## Data 
#
# The ecg_simulate function from the neurkit2 library allows generating an artificial ECG signal of a specified duration and sampling rate, using either the ECGSYN dynamical model (McSharry et al., 2003), which accurately reproduces cardiac waveforms, or a simpler model based on Daubechies wavelets.
#

# %%
fs=1000
T= 10
times= np.linspace(0, T, T * fs, endpoint=False)
ecg_signal= nk.ecg_simulate(duration=T, sampling_rate=fs, method="simple", noise = 0)
gaussian_noise= np.random.normal(0, 0.1, ecg_signal.shape)
ecg_signal += gaussian_noise

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(times, ecg_signal)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Simulated ECG Signal")
plt.show()

# %% [markdown]
# ### Denoising with filters

# %% [markdown]
# <div class="alert alert-success" role="alert">
#   <p><b>Question</b></p>
#   <p>
#     Plot the <b>log-spectrum</b> of the signal.<br>
#     From this plot:
#     <ul>
#       <li>Estimate the <b>noise amplitude</b>.</li>
#       <li>Estimate a suitable <b>cutoff frequency</b> for denoising the signal.</li>
#     </ul>
#   </p>
# </div>

# %%
import numpy as np
import matplotlib.pyplot as plt

fft_vals = rfft(ecg_signal)
freqs = rfftfreq(n=len(ecg_signal), d=1/fs)

N = T * fs 

# Amplitude en dB
log_dft = 10 * np.log10(np.abs(fft_vals)**2 / N)

# Plot
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(freqs, log_dft)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Log-Spectrum")
plt.show()


# %%
fc = 20 # Cutoff frequency in Hz
estimated_noise= np.mean(log_dft[freqs > 200])

# %% [markdown]
# We can use a low-pass Butterworth filter to denoise the signal.

# %%
from scipy.signal import butter, filtfilt

order = 4 # filter order

b, a = butter(order, fc , btype='low', analog=False, fs=fs)

# Apply the filter
filtered_signal = filtfilt(b, a, ecg_signal)

plt.figure(figsize=(12, 5))
plt.plot(times, ecg_signal, label="Original Signal", alpha=0.6)
plt.plot(times, filtered_signal, label="Filtered Signal (Butterworth low-pass)", linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("ECG Signal Denoising using Butterworth Low-Pass Filter : MSE = {:.4f}".format(np.mean((ecg_signal - filtered_signal)**2)))
plt.legend()
plt.show()

# %% [markdown]
# ### Sinusoidal model

# %% [markdown]
# In this section, we aim to reconstruct this signal using a simplified sinusoidal model.

# %%
fs=1000
T= 10
times= np.linspace(0, T, T * fs, endpoint=False)
ecg_signal= nk.ecg_simulate(duration=T, sampling_rate=fs, method="simple", noise=0.1)


fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(times, ecg_signal)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Simulated ECG Signal")
plt.show()


# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Compute and display the autocorrelation function of the signal.
#     </p> Propose a method to estimate the fundamental frequency and apply it to the signal</p>
# </div>

# %%
autocorr = acf(ecg_signal, nlags=fs*2)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(autocorr)
ax.set_title("Autocorrelation")
plt.show()

m0 = get_largest_local_max(autocorr, n_largest=1, order=fs//2)[1][0]
f0 = fs / m0
print(f"Estimated F0: {f0} Hz")

# %%

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Using the function below, apply the sinusoidal model to the signal for several values of order (e.g., 1, 3, 5, 10):</p>
#     <ul>
#         <li>For each order, plot both the <b>original signal</b> and the <b>reconstructed signal</b> on the same graph.</li>
#         <li>Compute and report the <b>mean squared error (MSE)</b> between the original and reconstructed signals for each order.</li>
#         <li>Briefly discuss how the reconstruction quality changes as the order increases.</li>
#     </ul>
# </div>
#

# %%
def sinusoidal_model(signal, timestamps, f0, order=2):
    """Fit a sinusoidal model to the signal."""
    X = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) * f0 * timestamps) for i in range(order)]
        + [np.cos(2 * np.pi * (i + 1) * f0 * timestamps) for i in range(order)]
    )
    coefs= np.linalg.lstsq(X, signal, rcond=None)[0]
    fitted_signal = X @ coefs
    return coefs, fitted_signal


# %%
order_list= [1, 3, 5, 10]
for order in order_list:
    _, fitted = sinusoidal_model(ecg_signal, times, f0, order)
    mse = np.mean((ecg_signal - fitted)**2)
    print(f"Order {order} MSE: {mse:.4f}")
    plt.figure(figsize=(15, 4))
    plt.plot(times, ecg_signal, label='Original')
    plt.plot(times, fitted, label=f'Reconstructed (Order {order})')
    plt.legend()
    plt.show()

# %% [markdown]
# ### Trend + Seasonality model

# %% [markdown]
# In this section we aim to reconstruct a signal with a trend+seasonality model.

# %%
fs=1000
T= 10
times= np.linspace(0, T, T * fs, endpoint=False)
ecg_signal= nk.ecg_simulate(duration=T, sampling_rate=fs, method="ecgsyn")


fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(times, ecg_signal)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Simulated ECG Signal")
plt.show()

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Estimate the fundamental frequency of the signal<p>
#     <p>Using the function below, apply the trend+seasonality model to the signal for a fixed seasonality order of 10 and for several values of trend order (e.g., 0, 1, 2, 5, 10).<p> 
# </div>

# %%
autocorr = acf(ecg_signal, nlags=fs*5)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(autocorr)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation Function of Simulated ECG Signal")
plt.show()

# %%
_, m0 = get_largest_local_max(autocorr, n_largest=1, order=fs//2)
m0=m0[0]
f0= fs / m0
f0


# %%
def trend_seasonality_model(signal, timestamps, f0, trend_order=2, seasonality_order=2):
    """Fit a trend + seasonality model to the signal."""
    seasonality = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) * f0 * timestamps) for i in range(seasonality_order)]
        + [np.cos(2 * np.pi * (i + 1) * f0 * timestamps) for i in range(seasonality_order)]
    )
    trend = np.column_stack(
        [timestamps**i for i in range(trend_order + 1)]
    )

    X = np.column_stack((trend, seasonality))
    coefs = np.linalg.lstsq(X, signal, rcond=None)[0]
    fitted_signal = X @ coefs
    return coefs, fitted_signal


# %%
trend_order_list = [0, 1, 2, 5, 10]

for order in trend_order_list:
    _, fitted = trend_seasonality_model(ecg_signal, times, f0, trend_order=order, seasonality_order=10)
    mse = np.mean((ecg_signal - fitted)**2)
    print(f"Trend Order {order} MSE: {mse:.4f}")
    plt.figure(figsize=(15, 4))
    plt.plot(times, ecg_signal, label='Original')
    plt.plot(times, fitted, label=f'Trend Order {order}')
    plt.legend()
    plt.show()

# %% [markdown]
# ## Signal prediction

# %% [markdown]
# In this section, we will use signal modeling techniques to perform signal prediction.

# %% [markdown]
# ## Data

# %%
X=pd.read_csv('nyc_taxi.csv')
X["timestamp"] = pd.to_datetime(X["timestamp"])

# %%
fig, ax = fig_ax()
_ = X.plot(x="timestamp", y="value", ax=ax)

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot the taxi count for October and for the 2014-10-12.</p>
# </div>

# %%

fig, ax = fig_ax()
X_october = X[(X["timestamp"] >= "2014-10-01") & (X["timestamp"] < "2014-11-01")]
_ = X_october.plot(x="timestamp", y="value", ax=ax, title="Taxi Count for October 2014")
plt.show()


fig, ax = fig_ax()
X_day = X[(X["timestamp"] >= "2014-10-12") & (X["timestamp"] < "2014-10-13")]
_ = X_day.plot(x="timestamp", y="value", ax=ax, title="Taxi Count for 2014-10-12")
plt.show()


# %% [markdown]
# ## Daily count
#
# In this tutorial, we are interested in the evolution in the **daily** count.
# To that end, we resample the original signal.

# %%
X["timestamp"] = pd.to_datetime(X["timestamp"])
daily_taxi_count = X.resample("1D", on="timestamp").sum()
daily_taxi_count_np = daily_taxi_count.to_numpy().squeeze()
fig, ax = fig_ax()
ax.plot(daily_taxi_count, "*-")
_ = ax.set_ylim(0)

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot the daily count in October 2014. What can you observe?</p>
# </div>

# %%
fig, ax = plt.subplots(figsize=(15, 4))
daily_taxi_count.loc["2014-10"].plot(ax=ax, marker="o", title="Daily Taxi Count October 2014")
ax.set_ylabel("Count")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()

# %% [markdown]
#  <div class="alert alert-success" role="alert">
#      <p><b>Question</b></p>
#      <p>What are the important periodicities in the original signal?</p>
#  </div>

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(daily_taxi_count_np, ax=ax[0], lags=60)
ax[0].set_title("ACF of Daily Taxi Count")


plot_pacf(daily_taxi_count_np, ax=ax[1], lags=60, method="ywm")
ax[1].set_title("PACF of Daily Taxi Count")
plt.tight_layout()
plt.show()


f, Pxx = periodogram(daily_taxi_count_np, scaling="density")
plt.figure(figsize=(12, 4))
plt.semilogy(f[1:], Pxx[1:], label="Periodogram (log scale)")
plt.xlabel("Frequency [cycles per day]")
plt.ylabel("Power spectral density")
plt.title("Power Spectrum of Daily Taxi Count")
plt.grid()
plt.tight_layout()
plt.show()


# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot or print the average count per hour of the day. Which time is the busiest?</p>
# </div>

# %%
X.groupby(X["timestamp"].dt.hour)["value"].mean().plot(kind='bar', figsize=(12, 4), title="Average Taxi Count by Hour")
plt.show()

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot or print the average count per day of the week. Which day is the busiest?</p>
# </div>

# %%
X.groupby(X["timestamp"].dt.day_name())["value"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).plot(kind='bar', figsize=(12, 4), title="Average Taxi Count by Day")
plt.show()

# %% [markdown]
# # Signal Prediction
#
# The objective is to predict the daily taxi for the next two weeks (14 days).
# First, we prepare the training and testing data.

# %%
n_samples_pred = 14  # predict the next 14 samples

# train/test split
signal_train, signal_pred = np.split(
    daily_taxi_count_np.astype(float), [-n_samples_pred]
)
n_samples = n_samples_train = signal_train.size

# scaling
scaler = StandardScaler().fit(signal_train.reshape(-1, 1))
signal_train = scaler.transform(signal_train.reshape(-1, 1)).flatten()
signal_pred = scaler.transform(signal_pred.reshape(-1, 1)).flatten()

# keep the indexes of train and test (for plotting mostly)
time_array_train, time_array_pred = np.split(
    np.arange(daily_taxi_count_np.size), [-n_samples_pred]
)
time_array = time_array_train
calendar_time_array = daily_taxi_count.iloc[time_array].index.to_numpy()

# plot
fig, ax = fig_ax()
ax.plot(time_array_train, signal_train, "-*", label="Train")
ax.plot(time_array_pred, signal_pred, "-*", label="To predict")
_ = plt.legend()

# %% [markdown]
# ## Trend
#
# Three trend estimation methods are tested:
#
# - constant trend,
# - linear trend,
# - polynomial trend.

# %%
fig, ax = fig_ax()
ax.plot(signal_train, label="Original")

level = signal_train.mean()  # should be zero
approx_trend = level * np.ones(signal_train.size)
ax.plot(approx_trend, label="Constant trend")
ax.set_title(f"MSE: {(signal_train-approx_trend).var():.2f}")
_ = plt.legend()

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot the best linear approximation and polynomial approximation (deg=13) of the signal (you may use <code>Polynomial.fit</code>). What are the associated MSE?</p>
# </div>

# %%
for order in [1,13]:
    p = Polynomial.fit(time_array_train, signal_train, order)
    trend = p(time_array_train)
    trend_pred = p(time_array_pred)
    mse = np.mean((signal_train - trend)**2)
    plt.figure(figsize=(15, 4))
    plt.plot(time_array_train, signal_train, label="Original")
    plt.plot(time_array_train, trend, label=f"Poly Order {order}")
    plt.plot(time_array_pred, trend_pred, label="Prediction")
    plt.title(f"Poly Order {order}, MSE: {mse:.4f}")
    plt.legend()
    plt.show()

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>In the previous cell, show the trend predicted by the polynomial fit in the next 14 samples. What do you conclude?</p>
# </div>

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>To conclude, which trend do you choose?</p>
# </div>

# %% [markdown]
#

# %% [markdown]
# ## Seasonality
#
# The seasonality is the periodical component in the signal at hand.
#
# **Finding the harmonic frequencies.**

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Give two ways to estimate the presence of seasonalities.</p>
# </div>
#

# %% [markdown]
#

# %% [markdown]
# The DFT is not a consistent estimator of the power spectral density.
# In practice, the periodogram (or any other variations) is prefered: the DTF is computed over several (possibly overlapping) windows and averaged.

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>What is the advantage of using the periodogram instead of the autocorrelation function to estimate the main periodicities in a signal?</p>
# </div>
#

# %%
autocorr = acf(signal_train, nlags= n_samples_train//2)
fig, ax = fig_ax()
ax.plot(autocorr)
ax.set_title("Autocorrelation function")
_ = ax.set_xlabel("Lag")
_, m0 = get_largest_local_max(autocorr, n_largest=1, order=5)
m0=m0[0]
print(f"Estimated seasonality period: {m0} samples")


# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Using the periodogram, estimate the two main frequencies present in the signal.</p>
# </div>
#

# %%
f, Pxx = periodogram(signal_train)
plt.figure(figsize=(15, 4))
plt.plot(f, Pxx)
plt.title("Periodogram")
plt.show()
# Estimate top 2 frequencies
top_indices = np.argsort(Pxx)[-2:]
f1, f2 = f[top_indices]
print(f"Dominant frequencies: {f1}, {f2}")

# %% [markdown]
# **Harmonic regression**
#
# In an harmonic regression (with two harmonic components), the signal is modelled as follows:
# $$
# y_t = \mu + A_1\cos(2\pi f_1 t + \phi_1) + A_2\cos(2\pi f_2 t + \phi_2) + \epsilon_t
# $$
#
# where $\mu, A, \phi\in\mathbb{R}$ must be estimated, the frequencies $f_1$ and $f_2$ are given, and $\epsilon_t$ is a white noise.

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>How can you rewrite this problem as a linear regression problem?</p>
# </div>

# %% [markdown]
#

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Code the harmonic regression with the two previously estimated frequencies and show the final fit, the residual signal and the MSE.</p>
# </div>
#

# %%
def harmonic_regression(signal, timestamps, f1, f2):
    """Fit a harmonic regression model to the signal."""
    X = np.column_stack([
        np.ones(len(timestamps)),
        np.cos(2 * np.pi * f1 * timestamps), np.sin(2 * np.pi * f1 * timestamps),
        np.cos(2 * np.pi * f2 * timestamps), np.sin(2 * np.pi * f2 * timestamps)
    ])
    coefs = np.linalg.lstsq(X, signal, rcond=None)[0]
    fitted_signal = X @ coefs
    return coefs, fitted_signal


# %%
coefs, fitted_train_signal = harmonic_regression(
    signal_train, time_array, f1=f1, f2=f2)

X_pred= np.column_stack((
    np.ones(time_array_pred.shape[0]),
    np.column_stack(
        [np.sin(2 * np.pi *  f1 * time_array_pred) ]
        + [np.cos(2 * np.pi  * f1 * time_array_pred) ]),
    np.column_stack(
        [np.sin(2 * np.pi *  f2 * time_array_pred) ]
        + [np.cos(2 * np.pi  * f2 * time_array_pred) ])  
))

fitted_pred_signal = X_pred @ coefs

fig,ax = fig_ax()
ax.plot(signal_train, label="Original")
ax.plot(fitted_train_signal, label="Fitted signal")
ax.plot(time_array_pred, signal_pred, "-*", label="To predict")
ax.plot(time_array_pred, fitted_pred_signal, "r--", label="Predicted trend")
ax.legend()
ax.set_title(f"Test_MSE: {(signal_pred - fitted_pred_signal).var():.2f}")

# %% [markdown]
# ## Studying the residual signal

# %% [markdown]
# ### A simulated example
#
# Simulate a MA(2) process and an AR(2) process.
# For each plot the autocorrelation and partial autocorrelation.

# %%
arparams = np.array([0.55, -0.25])  #
maparams = np.array([0.65, 0.35])

ar = np.r_[1, -arparams]  # add zero-lag and negate
ma = np.r_[1, maparams]  # add zero-lag


n_samples_simulated = 1000
ar2 = arma_generate_sample(ar, [1], n_samples_simulated)
ma2 = arma_generate_sample([1], ma, n_samples_simulated)

fig, ax = fig_ax()
ax.plot(ar2, label="AR(2)")
ax.plot(ma2, label="MA(2)")
_ = plt.legend()

# %%
fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize=(20, 4))
_ = plot_acf(ar2, ax=ax_0, title="Autocorrelation AR(2)")
_ = plot_pacf(ar2, ax=ax_1, title="Partial autocorrelation AR(2)")

# %%
fig, (ax_0, ax_1) = plt.subplots(1, 2, figsize=(20, 4))
_ = plot_acf(ma2, ax=ax_0, title="Autocorrelation MA(2)")
_ = plot_pacf(ma2, ax=ax_1, title="Partial autocorrelation MA(2)")

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>What would be a procedure to estimate the AR and MA order of a process?</p>
# </div>

# %%

# %% [markdown]
# ### Back to our problem

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Plot the autocorrelation and partial autocorrelation of the residual signal (without the constant and harmonic trend).</p>
# </div>

# %%
residual_signal = signal_train - fitted_train_signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(residual_signal, ax=ax1)
plot_pacf(residual_signal, ax=ax2)
plt.show()

# %% [markdown]
# Now, we fit an ARMA process on the residual signal.

# %%
ma_order = 7
ar_order = 8

res = ARIMA(residual_signal, order=(ar_order, 0, ma_order)).fit()
print(res.summary())

# %% [markdown]
# Using the fitted model, it is now possible to predict the value of the residual signal.

# %%
in_sample_pred = res.predict()
out_sample_pred = res.forecast(n_samples_pred)

fig, ax = fig_ax()
ax.plot(time_array_train, residual_signal, label="True residual")
ax.plot(time_array_train, in_sample_pred, label="In-sample prediction")
ax.plot(time_array_pred, out_sample_pred, label="Out-of-sample prediction")
_ = plt.legend()

# %% [markdown]
# ### Final prediction

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Make and plot the final prediction for the taxi count (not normalized) for the next two weeks, using the trend, seasonal and residual processes.</p>
#     <p>What do you conclude?</p>
# </div>

# %%

# %% [markdown]
# ### Stationarity checks

# %% [markdown]
# As was previously seen, the residual signal was not completely stationary since it still contained seasonal and low frequency components.
# To assess this intuition, several statistical tests exists. Two of the most well-known are:
#
# - the Dickey-Fuller test (H0: the signal has a unit root); 
# - the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test (H0: the signal is trend/level stationary vs H1:the signal has a unit root).
#
# Actually, they do not test for stationarity but for symptoms of non-stationarity.

# %%
adf_test(residual_signal)

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>What can you conclude from the test?</p>
# </div>

# %% [markdown]
# ## Manage complex trend and outliers with SSA
#
# With Singular Spectrum Analysis (SSA), it is possible to manage the low frequency trends and seasonal effects with the same procedure.
# SSA is often described as a "PCA for signals".
#
# Let $y = \{y_t\}_t$ denote a $T$-sample long univariate signal, and $L$ a window length.
# The trajectory matrix $X$ is formed by  $M$ lag-shifted copies of $y$, i.e.
#
# $$
# X:=
# \begin{bmatrix}
# y_1&y_2&y_3&\ldots&y_{L}\\
# y_2&y_3&y_4&\ldots&y_{L+1}\\
# y_3&y_4&y_5&\ldots&y_{L+2}\\
# \vdots&\vdots&\vdots&\ddots&\vdots
# \end{bmatrix}
# $$
#
#
# Now, write the Singular Value Decomposition (SVD) of $X$ is as follows:
#
# $$
# X = U\Sigma V^T = \sum_{i=1}^{L} X_i\quad\text{with}\quad X_i:= \sigma_i u_i v_i^T
# $$
#
# where $\sigma=\text{diag}(\sigma_1,\dots,\sigma_L)$ are the singular values sorted in descending order, $u_i$ and $v_i$ are respectively the associated left and right singular vectors corresponding to the columns of the orthogonal matrices $U$ and $V$.
# Each $X_i$ is itself a trajectory matrix.

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>For a signal of length $T$ and a window of length $L$, what are the dimensions of the trajectory matrix (number of rows and columns)?</p>
# </div>

# %% [markdown]
#

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>How can you go from a trajectory matrix to a signal?</p>
# </div>

# %% [markdown]
# Let us apply apply SSA on the signal.

# %%
window_shape = 14
trajectory_matrix = get_trajectory_matrix(signal_train, window_shape)

# %%
# SVD
u, eigenvals, vh = np.linalg.svd(trajectory_matrix, full_matrices=False)

# %% [markdown]
# NOTE: shape of signal depending on eigenvalue?

# %%
plt.plot(eigenvals, "-*")

# %%
ssa_decomposition = np.zeros((signal_train.size, window_shape))

for ind, (left, sigma, right) in enumerate(zip(u.T, eigenvals, vh)):
    ssa_decomposition.T[ind] = average_anti_diag(
        sigma * np.dot(left.reshape(-1, 1), right.reshape(1, -1))
    )

# %%
fig, ax_arr = plt.subplots(
    nrows=window_shape // 3 + 1,
    ncols=3,
    figsize=(20, 3 * (window_shape // 3 + 1)),
)


for ind, (component, ax) in enumerate(zip(ssa_decomposition.T, ax_arr.flatten())):
    ax.plot(component)
    ax.set_xlim(0, component.size)
    ax.set_ylim(-2, 2)
    ax.set_title(f"Component n°{ind}")

# %% [markdown] jupyter={"source_hidden": true}
# In pratice, the trend (a slowly varying component), the periodic components and noise are well separated by SSA.
#
# We can plot the successive reconstructions when adding one SSA component at a time.

# %% jupyter={"source_hidden": true}
fig, ax_arr = plt.subplots(
    nrows=window_shape // 3 + 1,
    ncols=3,
    figsize=(20, 3 * (window_shape // 3 + 1)),
)

reconstruction = np.zeros(signal_train.size)

for component, ax in zip(ssa_decomposition.T, ax_arr.flatten()):
    reconstruction += component
    ax.plot(signal_train)
    ax.plot(reconstruction)
    ax.set_xlim(0, reconstruction.size)
    ax.set_ylim(-5, 4)

# %% [markdown]
# **Grouping**
#
# Notice that several SSA components are very similar.
# Usually they are summed together to deacrease the dimension of the representation.
# This operation is called "grouping".

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>As in the previous tutorial, use a hierarchical clustering approach to group the SSA components together. (use the Euclidean distance and plot the associated dendogram.)</p>
# </div>

# %%
Z = hierarchy.linkage(pdist(ssa_decomposition.T), method="ward")
plt.figure(figsize=(10, 5))
hierarchy.dendrogram(Z)
plt.show()

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Report the groups that you found in the <tt>groups</tt> variable.</p>
# </div>

# %%
groups=[[0], [1, 2], [3, 4], list(range(5, window_shape))]

# %% [markdown]
# Let us plot each SSA group individually.

# %%
# grouping
grouped_ssa = np.zeros((signal_train.size, len(groups)))

for dim_ind, component_indexes in enumerate(groups):
    grouped_ssa.T[dim_ind] = np.take(ssa_decomposition, component_indexes, axis=-1).sum(
        axis=1
    )

fig, ax = fig_ax()
_ = ax.plot(grouped_ssa)

# %% [markdown]
# **Prediction**
#
# The SSA components are then individually extrapolated by fitting an autoregressive model.
# The extended components are summed to produce the forecast values.

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Make and plot the final prediction for the taxi count (not normalized) for the next two weeks, by fitting an AR(8) process (use <tt>AutoReg(signal_train, lags=8)</tt>) to each SSA component and summing the forecasts.</p>
# </div>

# %%
n_samples_train

# %%
ssa_preds = np.zeros((len(groups), n_samples_pred))
for i, indices in enumerate(groups):
    component = grouped_ssa[:, i]
    model = AutoReg(component, lags=8).fit()
    ssa_preds[i] = model.forecast(steps=n_samples_pred)
total_ssa_pred = ssa_preds.sum(axis=0)

plt.figure(figsize=(15, 4))
plt.plot(time_array_pred, signal_pred, label="True")
plt.plot(time_array_pred, total_ssa_pred, label="SSA Prediction")
plt.legend()
plt.show()
# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Conclude. (Is it better or worse than the previous approach? What can we do to improve the results? What is the limitation?)</p>
# </div>

# %%

# %% [markdown]
# # Outliers detection/removal

# %%
original_calendar_time_array = X.timestamp.to_numpy()
original_taxi_count_np = X.value.to_numpy()

# %%
daily_taxi_count_np = daily_taxi_count.to_numpy().squeeze()
calendar_time_array = daily_taxi_count.index.to_numpy()
n_samples = daily_taxi_count_np.size
fig, ax = fig_ax()
ax.plot(daily_taxi_count, "*-")
_ = ax.set_ylim(0)

# %%
quantile_threshold_low, quantile_threshold_high = 0.01, 0.99

fig, ax = fig_ax()
_ = ax.hist(daily_taxi_count_np, 20)

threshold_low, threshold_high = np.quantile(
    daily_taxi_count_np, [quantile_threshold_low, quantile_threshold_high]
)

_ = ax.axvline(threshold_low, ls="--", color="k")
_ = ax.axvline(threshold_high, ls="--", color="k")

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>In the previous cell, modify <tt>quantile_threshold_low</tt> and <tt>quantile_threshold_high</tt> to only exclude outliers.</p>
# </div>

# %% [markdown]
# Plot the outliers directly on the signal.

# %%
fig, ax = fig_ax()
ax.plot(calendar_time_array, daily_taxi_count_np, "*-", label="Daily taxi count")

outlier_mask = (daily_taxi_count_np < threshold_low) | (
    daily_taxi_count_np > threshold_high
)

ax.plot(
    calendar_time_array[outlier_mask],
    daily_taxi_count_np[outlier_mask],
    "*",
    label="Outliers",
)

plt.legend()
_ = ax.set_ylim(0)

# %% [markdown]
# <div class="alert alert-success" role="alert">
#     <p><b>Question</b></p>
#     <p>Apply a median filter to remove the outliers present in the signal.</p>
# </div>

# %%
cleaned_signal = medfilt(daily_taxi_count_np, kernel_size=3)
fig, ax = fig_ax()
ax.plot(calendar_time_array, daily_taxi_count_np, label="Original", alpha=0.5)
ax.plot(calendar_time_array, cleaned_signal, label="Median Filtered", color='red')
plt.legend()
plt.show()
