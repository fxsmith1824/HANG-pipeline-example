# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:38:30 2022

This version is intended to follow the application of the CIAC algorithm to 
finally check components, reject eyeblinks/movements, etc. and do a final
epochRejection.

Notes:
- OT0643 has the weirdest pattern I've ever seen - CIAC completely ineffective
but so is manual removal, it seems. Drop participant?
- OT0742 only has 32 recorded electrodes. Drop.
- OT0653 doesn't seem to have a CI artifact at all...
- OT0641 has... 4 epochs? DROP

- OT0704 CI artifact was beautifully captured by basically 1 primary component
- OT0648 is one example of the aggressive parameters catching an obvious CI 
artifact that was missed on first pass (ICA013)

@author: Francis
"""

import mne
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Lazily changed final epochs.info['description'] to be 'Final threshold'
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
    epochs.info['description'] = epochs.info['description'] + 'Final epoch rejection threshold: ' + str(epochThreshold) + '.'

#######################################
# If reprocess_data is True, change file saving overwring to be True
# And skip the check for already created -epo.fif files for each sID
reprocess_data = False
# Do we want to use CIAC pre-processed ICA files?
ciac_preprocessed = True

cwd = os.getcwd()
ICA_folder = '2_ICA_set'
postICA = '3_mne_epochs_after_rejection'

try:
    with open('lab_members.pickle', 'rb') as file:
        lab_members = pickle.load(file)
except:
    lab_members = dict()

lab_member = input('Please enter your first and last initial with no spaces (e.g. FS for Francis Smith):\n')
if lab_member not in lab_members:
    print('Lab member ' + lab_member + ' is not in stored file - adding lab member')
    lab_members[lab_member] = []
already_processed = [file[0:6] for file in os.listdir(postICA) if file.endswith('-epo.fif')]
already_processed = [file for file in already_processed if already_processed.count(file) > 1]
self_fnames = lab_member + '-epo.fif'
self_already_processed = [file[0:6] for file in os.listdir(postICA) if file.endswith(self_fnames)]
lab_members[lab_member] = self_already_processed
print('----------')
print('You have processed ' +str(len(lab_members[lab_member])) + ' in previous sessions.')
print('----------')

with open('lab_members.pickle', 'wb') as file:
    pickle.dump(lab_members, file)

# Get list of all subjects who have had ICA run
all_sIDs = list(set([subject[:6] for subject in os.listdir(ICA_folder) if subject.endswith('-ica.fif')]))

while True:
    # Get list of all subjects who have had two people process already
    already_processed = [file[0:6] for file in os.listdir(postICA) if file.endswith('-epo.fif')]
    already_processed = [file for file in already_processed if already_processed.count(file) > 1]

    # Also get list of all subjects this lab member has processed
    self_fnames = lab_member + '-epo.fif'
    self_already_processed = [file[0:6] for file in os.listdir(postICA) if file.endswith(self_fnames)]

    # Find subjects who have ICA run but have not been processed twice
    sIDs = [file for file in all_sIDs if file not in already_processed]
    sIDs = [file for file in sIDs if file not in self_already_processed]
    sID = sIDs[0]
    fname = sID + '-epo.fif'
    epochs = mne.read_epochs(os.path.join(cwd, ICA_folder, fname))
    epochs.load_data()
    if ciac_preprocessed:
        fname_ica = sID + '-CIAC-ica.fif'
    else:
        fname_ica = sID + '-ica.fif'
    ica = mne.preprocessing.read_ica(os.path.join(cwd, ICA_folder, fname_ica))
    
    epochs2 = epochs.copy()
    print('Current subject: ' + sID)
    ica.plot_components(inst=epochs2, psd_args=dict(fmin=0, fmax=60))
    print('------')
    print('Currently excluded components: ', ica.exclude)
    print(str(len(ica.exclude)) + ' components currently marked for exclusion.')
    ica.plot_overlay(inst=epochs2.average())
    
    breakpoint()
    # NOTE: YOU CAN CLICK ICA COMPONENT NAMES ON MULTIPLOT WINDOW TO MARK THEM
    # FOR REMOVAL (light gray font color) or INCLUSION (black font color)
    ica_ex_string = input("What components do you wanna remove? Use comma for multiple components.\n")
    if len(ica_ex_string)>0:
        ica.exclude += [int(i) for i in ica_ex_string.split(',')]
    ica.apply(epochs2)

    plt.close('all')

    epochRejection(epochs2, baseline=(-0.2,0))
    
    ica_fname = os.path.join(cwd, postICA, sID + '-' + lab_member + '-ica.fif')
    ica.save(ica_fname, overwrite=reprocess_data)
    fname = os.path.join(cwd, postICA, sID + '-' + lab_member + '-epo.fif')
    epochs2.save(fname, overwrite=reprocess_data)
    
    epochs.apply_baseline((-0.2,0))
    epochs2.apply_baseline((-0.2,0))
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6.))
    fig.tight_layout(pad=5.0)
    epochs.average().plot(axes=axes[0], spatial_colors=True)
    epochs2.average().plot(axes=axes[1], spatial_colors=True)
    fig_fname = os.path.join(cwd, postICA, sID + '-' + lab_member + '_BeforeAfter.pdf')
    plt.savefig(fig_fname)
    plt.close()
    
    continue_processing = input('Would you like to process the next subject? (y/n):\n')
    if continue_processing == 'n':
        break