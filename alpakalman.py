import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
from enum import Enum
import scipy
from pykalman import KalmanFilter, UnscentedKalmanFilter

##import alpaca_backtrader_api
##import backtrader as bt
NY = 'America/New_York'


class TimeFrame(Enum):
    Day = "1Day"
    Hour = "1Hour"
    Minute = "1Min"
    Sec = "1Sec"


symb = 'AAPL'

load_dotenv()
APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY')
APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL')
APCA_API_DATA_URL = os.environ.get('APCA_API_DATA_URL')
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, 'v2')
start=pd.Timestamp('2020-01-01', tz=NY).isoformat()
end=pd.Timestamp('2020-06-01', tz=NY).isoformat()
bars0 = api.get_barset(symb, timeframe='day', \
       start=start, end=end, limit = 1000).df
bars0.index.name = 'date'
##bars = pd.DataFrame(data=bars0[symb][['open', 'high', 'low', 'close']])
bars = pd.DataFrame(data=bars0[symb][['close']])


print(bars)

##bars[symb, 'open'].plot()


#phi: state transition model
#H: observation model
#Q: covariance of the process noise
startbar = bars.iloc[0:2]
x_01 = startbar.to_numpy()
x_0 = x_01[0]
Q = np.cov(x_01, rowvar=False)
mom = [1]

#R: covariance of observation noise
##R = 0.005*np.identity(len(x_0))
##print(Q)
##print('Q eigenvalues:')
##print(np.linalg.eigvals(Q))
##print(R)
##print('R eigenvalues:')
##print(np.linalg.eigvals(R))



kf = KalmanFilter(transition_matrices = mom,
                  observation_matrices = [1],
                  initial_state_mean = x_0,#bars.iloc[0],
                  initial_state_covariance = Q,
                  observation_covariance=1,
                  transition_covariance=Q)



loglikelihoods = np.zeros(10)
for i in range(len(loglikelihoods)):
    kf = kf.em(X=bars.to_numpy(), n_iter=5)
    loglikelihoods[i] = kf.loglikelihood(bars.to_numpy())

plt.figure()
plt.plot(loglikelihoods)
plt.xlabel('EM iteration')
plt.ylabel('Log Likelihood')
plt.show()

# Use the observed values of the price to get a rolling mean
state_means, state_covs = kf.filter(bars)
state_means = pd.Series(state_means.flatten(), index=bars.index)
state_covs = pd.Series(state_covs.flatten(), index=bars.index)
smooth_state_means, smooth_state_covs = kf.smooth(bars)
smooth_state_means = pd.Series(smooth_state_means.flatten(), index=bars.index)
smooth_state_covs = pd.Series(smooth_state_covs.flatten(), index=bars.index)


# Compute the rolling mean with various lookback windows
mean30 = bars.rolling(window = 3).mean()
mean60 = bars.rolling(window = 15).mean()
mean90 = bars.rolling(window = 30).mean()

# Plot original data and estimated mean

##plt.plot(mean60)
##plt.plot(mean90)
fig, ax1 = plt.subplots(1, 1, sharex=True)
ax1.set_title('Kalman Filter Estimate')
ax1.plot(state_means)
ax1.plot(smooth_state_means)
##ax2.plot(state_covs)
##ax2.plot(smooth_state_covs)
ax1.plot(bars)
ax1.legend(['Kalman Estimate', 'Smooth Kalman', 'True'])
plt.xlabel('Day')
plt.ylabel('Price');
plt.xlim((bars.index[0],bars.index[-1]))
fig.show()

print('done with step 1')


#predicting tomorrow's closing price
print('last state mean:')
print(state_means[-1])
print('predicted tm mean:')
predicted_state_mean = (
        np.dot(kf.transition_matrices, state_means[-1])
        + kf.transition_offsets
    )
print(predicted_state_mean)


# Estimate the state without using any observations.  This will let us see how
# good we could do if we ran blind.
n_dim_state = len(x_0)
n_timesteps = len(bars.iloc[:,0])
blind_state_estimates = np.zeros((n_timesteps, n_dim_state))
for t in range(n_timesteps - 1):
    if t == 0:
        blind_state_estimates[t] = kf.initial_state_mean
    blind_state_estimates[t + 1] = (
      np.dot(kf.transition_matrices, blind_state_estimates[t])
      + kf.transition_offsets
    )




em=0
if em==1:
    loglikelihoods = np.zeros(10)
    for i in range(len(loglikelihoods)):
        kf = kf.em(X=bars.to_numpy(), n_iter=5)
        loglikelihoods[i] = kf.loglikelihood(bars.to_numpy())

    # Estimate the state without using any observations.  This will let us see how
    # good we could do if we ran blind.
    n_dim_state = len(x_0)
    n_timesteps = len(bars.iloc[:,0])
    blind_state_estimates = np.zeros((n_timesteps, n_dim_state))
    for t in range(n_timesteps - 1):
        if t == 0:
            blind_state_estimates[t] = kf.initial_state_mean
        blind_state_estimates[t + 1] = (
          np.dot(kf.transition_matrices, blind_state_estimates[t])
          + kf.transition_offsets
        )

    # Estimate the hidden states using observations up to and including
    # time t for t in [0...n_timesteps-1].  This method outputs the mean and
    # covariance characterizing the Multivariate Normal distribution for
    #   P(x_t | z_{1:t})
    filtered_state_estimates = kf.filter(bars.to_numpy())[0]

    # Estimate the hidden states using all observations.  These estimates
    # will be 'smoother' (and are to be preferred) to those produced by
    # simply filtering as they are made with later observations in mind.
    # Probabilistically, this method produces the mean and covariance
    # characterizing,
    #    P(x_t | z_{1:n_timesteps})
    smoothed_state_estimates = kf.smooth(bars.to_numpy())[0]

    # Draw the true, blind,e filtered, and smoothed state estimates for all 5
    # dimensions.
    plt.figure(figsize=(16, 6))
    lines_true = plt.plot(bars.index, bars.to_numpy()[:,0], linestyle='-', color='b')
    lines_blind = plt.plot(bars.index, blind_state_estimates[:,0], linestyle=':', color='m')
    lines_filt = plt.plot(bars.index, filtered_state_estimates[:,0], linestyle='--', color='g')
    lines_smooth = plt.plot(bars.index, smoothed_state_estimates[:,0], linestyle='-.', color='r')
    plt.legend(
        (lines_true[0], lines_blind[0], lines_filt[0], lines_smooth[0]),
        ('true', 'blind', 'filtered', 'smoothed')
    )
    plt.xlabel('time')
    plt.ylabel('state')
    plt.xlim((bars.index[0], bars.index[-1]))
    plt.show()
    print('done with step 2')
