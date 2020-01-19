from scipy.stats import pearsonr
# import tensorflow as tf
from scipy.stats.mstats import zscore
from sklearn.linear_model import LogisticRegression  # L2
import operator
import itertools
from itertools import combinations
from sklearn.externals import joblib
import collections
from functools import wraps
from scipy.stats import ks_2samp
from cmlreaders import CMLReader, get_data_index
from sklearn.metrics import roc_auc_score
import json
from collections import defaultdict
import glob, os
import numpy as np 

from ptsa.data.filters import morlet
from statsmodels.tsa.api import VAR



def construct_model_based_connectivity(event_type, reader, pairs, BUFFER, freqs, EPSILON, window_size): 
    sess_events = reader.load('task_events')
    events = sess_events[sess_events.type == event_type]
    rel_start = 0 
    rel_stop = 0
    if event_type == 'WORD':
        rel_stop = 1366
    if event_type == 'COUNTDOWN_START': 
        countdown_end_events = sess_events[sess_events.type == 'COUNTDOWN_END']
        countdown_times = []
        for i in np.arange(len(countdown_end_events)):
            countdown_times.append(countdown_end_events.iloc[i]['mstime'] - events.iloc[i]['mstime'])

        rel_stop = np.min(countdown_times)
    else:
        rel_stop = 1366

    events_eeg = reader.load_eeg(events, rel_start=rel_start, rel_stop=rel_stop, scheme=pairs)
    events_eeg = events_eeg.to_ptsa()
    events_eeg = events_eeg.filtered(freq_range=[58.0,62.0])
    events_eeg.dims
    events_eeg = events_eeg.add_mirror_buffer(BUFFER)
    wf = morlet.MorletWaveletFilter(events_eeg, freqs = freqs)
    power_wavelet, phase_wavelet = wf.filter()
    power_wavelet = power_wavelet.remove_buffer(BUFFER)
    power_wavelet = power_wavelet.transpose('channel', 'event', 'time', 'frequency')
    power_wavelet = np.log10(power_wavelet + EPSILON)
    n_times = power_wavelet.shape[2]
    intervals = np.array_split(np.arange(n_times), int(n_times/window_size))
    power_wavelet_aggregate = np.zeros(shape = list(power_wavelet.shape[:2]) + [len(intervals)])
    for i in np.arange(len(intervals)):
        power_wavelet_aggregate[:,:,i] = power_wavelet[:,:,intervals[i],0].mean('time')

    dims =  power_wavelet_aggregate.shape 
    power_wavelet_aggregate = power_wavelet_aggregate.reshape(dims[0], dims[1]*dims[2])
    model = VAR(power_wavelet_aggregate[:,:].T)
    results = model.fit(maxlags = 1)
    conn_mat = results.coefs[0,:,:]
    return conn_mat
