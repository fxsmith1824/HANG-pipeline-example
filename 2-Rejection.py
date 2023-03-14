# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:40:08 2022

This step requires user input, but should only do histograms

Designed epochRejection and channelRejection functions to do the histogram 
plotting

For now, just define function here. It will be incorporated into the HANG
Pipeline files later.

To recreate the final list of dropped epochs, can use something like this
whichEpochs = [i for i in range(len(epochs.drop_log)) if len(epochs.drop_log[i]) > 0]

@author: Francis
"""

import mne
import os
import pandas as pd
import matplotlib.pyplot as plt

def epochRejection(epochs, baseline=(-0.2,0)):
    '''
    Function to implement the same histogram procedure as used in the MATLAB
    processing pipeline. This function first creates a copy of the epochs
    object and applies a baseline (see baseline parameter for more information)
    after which a histogram is generated of the maximum voltage (absolute 
    value) observed in each epoch across all non-excluded channels.
    
    Press "c" to exit debugging and continue to picking a rejection threshold.
    
    Because MNE's Epochs drop() method operates in place, this function does 
    not actually return any output.
    
    NOTE: This function is meant to be applied to data to which a baseline has
    not yet been applied (e.g. baseline=None during initial epoching). It has 
    not yet been tested with data which have already been baselined.
    
    NOTE2: Have gone back and forth on the use of breakpoint() - which opens
    the Python debugger much like the MATLAB script drops into debugging to
    view plots before continuing - OR using plt.pause(1) - which pauses the
    script long enough to generate a viewable plot and then continues the 
    script (which pauses as it waits for input about the threshold value). I 
    feel that using the debugging mode is less professional, but it does lead
    to an interactive plot in which the x/y values of the mouse position are
    constantly reported. This makes picking a voltage threshold easier. The 
    pause function allows the plot to be viewed but it is less interactive and
    the x/y coordinates are not reported. Sticking with breakpoint() for now.
    To substitue plt.pause(1) simply replace the breakpoint() line.
    
    TO DO: 
        - Consider returning a variable of "whichEpochs" rather than outright
        dropping epochs within function. The MATLAB script doesn't actually
        drop any epochs before the channel histogram (or so it seems) as the
        second epoch histogram looks identical to the first even after picking
        a threshold. Ask Inyong if rejecting epochs before the initial channel
        histogram is acceptable. At the very least, rejecting before second
        histogram leads to better x-axis resolution for second plot.
        - Test how this function acts with data that are already baselined. The
        apply_baseline() method of Epochs doesn't SEEM to do anything on data
        that are already loaded with a previous baseline (as intended) so this
        should simply plot using that baseline, but this has not been tested
        yet.
        - It seems that when max voltage is significantly greater than 1000 to
        1500 across channels for several epochs, this is often due to a single
        bad channel. Consider adding recommendation to skip first epochReject
        when this is the case to eliminate bad channel first. Alternatively, 
        consider ALWAYS looking at channelRejection first (leading to a 
        sequence of channelReject, epochReject, channelReject, epochReject).

    Parameters
    ----------
    epochs : mne.epochs.EpochsFIF CLASS
        The epochs class to be used in processing.
    baseline : TUPLE, optional
        A tuple of the initial and final time to be used as the baseline. In 
        the current Python processing pipeline, baselining is only performend
        after ICA, but plotting a histogram of non-baselined voltages results 
        in data that are hard to interpret. This should be the same as the
        desired baseline for later processing. The default is (-0.2,0).

    Returns
    -------
    None.

    '''
    channels = [ch for ch in epochs.info['ch_names'] if ch not in epochs.info['bads']]
    epochs_copy = epochs.copy()
    epochs_copy.apply_baseline(baseline)
    df = epochs_copy.to_data_frame()
    epochMax = []
    del epochs_copy
    for epoch in df['epoch'].unique():
        voltage = df.loc[df['epoch']==epoch]
        voltage = voltage[channels].abs()
        maxValue = voltage[channels].max(axis=1).max(axis=0)
        result = [epoch, maxValue]
        epochMax.append(result)
    epochMax = pd.DataFrame(data=epochMax, columns=['Epoch', 'MaxValue'])
    q1 = epochMax['MaxValue'].quantile(.25)
    q3 = epochMax['MaxValue'].quantile(.75)
    iqr = q3-q1
    cutoff = int(q3+1.5*iqr)
    plt.hist(epochMax['MaxValue'])
    title_text = 'Automatic suggested threshold (dotted line): ' + str(cutoff)
    plt.title(title_text)
    plt.axvline(x=cutoff,linestyle='dotted',color='black')
    plt.show()
    breakpoint()
    epochThreshold = input('What threshold for rejecting epochs?\n')
    plt.close()
    whichEpochs = epochMax[epochMax.iloc[:,1] > int(epochThreshold)]
    epochs.drop(whichEpochs.index[:].tolist())
    if not epochs.info['description']:
        epochs.info['description'] = 'Initial epoch rejection threshold: ' + str(epochThreshold) + '.  '
    else:
        epochs.info['description'] = epochs.info['description'] + 'Second epoch rejection threshold: ' + str(epochThreshold) + '.'

def channelRejection(epochs, baseline=(-0.2,0)):
    '''
    Function to implement the same histogram procedure as used in the MATLAB
    processing pipeline. This function first creates a copy of the epochs
    object and applies a baseline (see baseline parameter for more information)
    after which a histogram is generated of the maximum voltage (absolute 
    value) observed in each epoch across all non-excluded channels.
    
    Press "c" to exit debugging and continue to picking a rejection threshold.
    
    Because MNE's Epochs drop() method operates in place, this function does 
    not actually return any output.
    
    See epochRejection() function for further notes on potential changes / TODO

    Parameters
    ----------
    epochs : TYPE
        DESCRIPTION.
    baseline : TYPE, optional
        DESCRIPTION. The default is (-0.2,0).

    Returns
    -------
    None.

    '''
    channels = [ch for ch in epochs.info['ch_names'] if ch not in epochs.info['bads']]
    epochs_copy = epochs.copy()
    epochs_copy.apply_baseline(baseline)
    df = epochs_copy.to_data_frame()
    del epochs_copy
    channelMax = []
    for channel in channels:
        temp = df[channel].abs()
        maxValue = temp.max()
        result = [channel, maxValue]
        channelMax.append(result)
    channelMax = pd.DataFrame(data=channelMax, columns=['Channel', 'MaxValue'])
    plt.hist(channelMax['MaxValue'])
    plt.title('Choose a max voltage for rejecting channels')
    plt.show()
    breakpoint()
    channelThreshold = input('What treshold for rejecting channels?\n')
    whichChannels = channelMax[channelMax.iloc[:,1] > int(channelThreshold)]
    badChannel = whichChannels['Channel'].tolist()
    print('The threshold you chose (' + channelThreshold + ') will result in the following channels being removed:')
    print(badChannel)
    proceed = input('Proceed? (y/n)\n')
    while proceed != 'y':
        channelThreshold = input('What threshold for rejecting channels?\n')
        whichChannels = channelMax[channelMax.iloc[:,1] > int(channelThreshold)]
        badChannel = whichChannels['Channel'].tolist()
        print('The threshold you chose (' + channelThreshold + ') will result in the following channels being removed:')
        print(badChannel)
        proceed = input('Proceed (y/n)\n')
    plt.close()
    for channel in badChannel:
        epochs.info['bads'].append(channel)
    epochs.info['description'] = epochs.info['description'] + 'Channel rejection threshold: ' + str(channelThreshold) + '.  '

#######################################
# If reprocess_data is True, change file saving overwring to be True
# And skip the check for already created -epo.fif files for each sID
reprocess_data = False

cwd = os.getcwd()
epochsFolder = '1_epochs_w_excluded_channel_info'
ICA_folder = '2_ICA_set'
postICA = '3_epochs_after_rejection'

event_dict = {'female/HighSNR': 2816, 'female/LowSNR': 3328,
              'male/HighSNR': 3072, 'male/LowSNR': 3584,
              'response/correct':25600, 'response/incorrect':12800}

sIDs = [subject[:6] for subject in os.listdir(epochsFolder) if subject.endswith('-epo.fif')]

# Check which sIDs already have -epo.fif files in epochsFolder
if reprocess_data:
    already_processed = []
else:
    already_processed = [file[0:6] for file in os.listdir(ICA_folder) if file.endswith('-epo.fif')]

for sID in sIDs:
    if 'noEEG' in sID:
        continue
    elif sID in already_processed:
        continue
    else:
        fname = sID + '-epo.fif'
        path = os.path.join(cwd, epochsFolder, fname)
        epochs = mne.read_epochs(path)
            
        # Plot histogram of epoch max voltage values
        epochRejection(epochs, baseline=(-0.2,0))
        
        # Plot histogram of channel max voltage values
        channelRejection(epochs, baseline=(-0.2,0))
    
        # Plot histogram of epoch max voltage levels - second time
        epochRejection(epochs, baseline=(-0.2,0))
        
        fname_preICA = os.path.join(cwd, ICA_folder, sID + '-epo.fif')
        epochs.save(fname_preICA, overwrite=reprocess_data)