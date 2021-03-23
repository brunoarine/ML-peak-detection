#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import stats
import gzip
import pickle

def generate_new_spectrum(old_spectrum, rate=1):    
    # auxiliary variable
    old_spectrum_ravel = [] 
    # number of channels
    m = len(old_spectrum) 
    # total number of counts
    total_count = np.sum(old_spectrum)    
    for ch in range(m):        
        old_spectrum_ravel.extend([ch] * old_spectrum[ch])           
    old_spectrum_ravel = np.array(old_spectrum_ravel)
    new_spectrum_ravel = np.random.choice(old_spectrum_ravel, size=int(total_count*rate))   
    new_spectrum = np.bincount(new_spectrum_ravel,minlength=m)
    return new_spectrum
        
def new_peak(array, count, loc, fwhm):
    '''
    Generates counts in random locations according to a gaussian probability.
    '''        
    sigma = fwhm/2.355
    for i in range(count):
        channel = int(round(stats.norm.rvs(loc, sigma),0 ))
        array[channel] += 1
    return array

def new_bg(array, count_per_channel):
    '''
    Generates counts in random locations according to a uniform probability.
    '''    
    m = len(array)
    return array + np.random.poisson(count_per_channel, size=m)


def simulated_spectra(iter=100, bg_range=(40,41), snr_db_range=(5,6), include_dummy=False, dummy_pos=35, dummy_prob=0.05):
    X = np.zeros((iter, 100))
    y = np.zeros(iter)
    dummy = np.zeros(iter)
    list_peakareas = []
    for i in range(iter):
        snr_db = np.random.randint(snr_db_range[0], snr_db_range[1])        
        snr = 10**(snr_db/10.)
        bg = np.random.randint(bg_range[0], bg_range[1])      
        
        peakarea = int(0.5*(snr**2 + np.sqrt(40*bg*snr**2 +snr**4)))
        mainpeak = np.random.uniform(0.0,1.0) <= 0.5
        dummypeak = np.random.uniform(0.0,1.0) <= dummy_prob
        x = np.zeros(100)
        x = new_bg(x,bg)
        if mainpeak:
            new_peak(x, count=peakarea, loc=50, fwhm=3)            
            y[i] = 1
            list_peakareas.append(peakarea)
        if dummypeak:
            # Cenario A (nao tem dummy peak)
            if include_dummy:
                new_peak(x, count=np.random.randint(60,100), loc=dummy_pos, fwhm=3)
            dummy[i] = 1
        X[i,:] = x
    return X, y, dummy