# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:05:55 2022

Epoching for ISNT data including auto-generated metadata with response times,
correct/incorrect information as well as manually added target word information
derived from .mat results file for each participant.

Adding bad channels information in this step (unlike in Longitudinal Analyses)
to comply with previous MATLAB processing steps.

UPDATE: 9/8/2022
Per Inyong's advice, changing filter length to 256 samples (sample frequency / 8)
MNE built in filtering function will likely have issues with the current l_freq
and h_freq parameters. Test out and report back.

Designed custom implementation of BPF.m called HANG_BFP.py

For now, just define function here. It will be incorporated into the HANG
Pipeline files later.

NOTE: raw.info['highpass'] and raw.info['lowpass'] can technically be set
manually to reflect our custom BPF function, but this will lead to errors in 
later MNE versions. Maybe implement our filter in MNE later? For now, raw.info
on high and low pass will simply be incorrect (0 and 256).

Also only downsampling to 512 Hz instead of 256 Hz per advice.

NOTE: If epoched data is a different length than the .mat file has target_word
information for, metadata is not saved. Will need to check manually which
epochs have metadata later.

NOTE: At *least* 3 patients had EX1 sensor replacing a normal sensor.
OT0651, OT0675, OT0779. I did not account for this on initial processing, and
will need to either re-process these or at least edit the info['bads'] and
ch_types information to correct this. - 9/20/2022

UPDATE: Per discussions with Inyong, changing this part of the pipeline to
use MNE Python's filter parameters and using an aggressive highpass filter 
limit for the purposes of ICA. Later pipeline (step 4 or new step 5) will need
to read in raw data again, use final filter parameters, re-epoch, and then use
the saved epochs after step 2 (rejection) to manually drop the same epochs 
again.

@author: Francis
"""

import os
import mne
from scipy.io import loadmat
from scipy.signal import firwin
from numpy import array, flipud
from mne.filter import _overlap_add_filter as fftfilt

def BPF(data, fs, norder, cf1, cf2):
    '''
    Implement a version of the BPF.m function for Python

    Parameters
    ----------
    data : ARRAY
        The data to be bandpass filtered.
    fs : INT
        The frequency at which data were sampled.
    norder : INT
        The n-th order for the filter, to generate n+1 numtaps.
    cf1 : INT
        The lower cutoff frequency for the bandpass filter.
    cf2 : INT
        The upper cutoff frequency for the bandpass filter.

    Returns
    -------
    An array of the bandpass filtered data.

    '''
    
    Ny = fs/2
    cutoffs = array([cf1,cf2])
    cutoffs = cutoffs / (Ny)

    filter_design = firwin(numtaps=norder+1, cutoff=cutoffs, pass_zero='bandpass')
    
    y = flipud(fftfilt(flipud(fftfilt(data, filter_design)), filter_design))
    return y

#######################################
# If reprocess_data is True, change file saving overwring to be True
# And skip the check for already created -epo.fif files for each sID
reprocess_data = True

epochsFolder = '1_epochs_w_excluded_channel_info'
cwd = os.getcwd()

montage = mne.channels.montage.read_dig_fif('Biosemi64median206subjects10percentLarger_dig.fif')
montage.ch_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32']

# Normally, trigger values are MATLAB VALUE * 256
event_dict = {'female/HighSNR': 2816, 'female/LowSNR': 3328,
              'male/HighSNR': 3072, 'male/LowSNR': 3584,
              'response/correct':25600, 'response/incorrect':12800}

# For some reason, some subjects' trigger values are encoded differently... 
# This alt_event dict will be used if that is detected - unclear how that will 
# affect later processing at this time. Seems to be based on 65280 which is
# 256*255 but then scales linearly so MATLAB VALUE + 65280
alt_event_dict = event_dict.copy()
for key, value in alt_event_dict.items():
    alt_event_dict[key] = int(value/256 + 65280)

removed_channels = {'OT0421': ['A16,A23,A24'],
                    'OT0431': ['A16,A23,A24,B21,B28,B29'],
                    'OT0432': ['A16,A23,A24,B10,B21'],
                    'OT0453': ['B21,B28,B29'],
                    'OT0496': ['A16,A23,A24'],
                    'OT0504': ['B21,B28,B29'],
                    'OT0515': ['A16,A23,A24,B20,B21,B29'],
                    'OT0517': ['A16,A23,A24,B20,B21,B29'],
                    'OT0518': ['A15'],
                    'OT0522': ['A16,A23,A24'],
                    'OT0524': ['B21,B28,B29'],
                    'OT0525': ['A15,A16,A22,A23,A24'],
                    'OT0527': ['A16,A23,A24'],
                    'OT0529': ['A16,A23,A24'],
                    'OT0539': ['A15,B21,B28,B29'],
                    'OT0544': ['A16,A23,A24'],
                    'OT0553': ['B20,B21,B29'],
                    'OT0559': ['A15,A16,A17,A22,A23,A24,A25'],
                    'OT0565': ['B21,B28,B29,A10,A18'],
                    'OT0567': ['B21,B28,B29,B25'],
                    'OT0570': ['B21,B28,B29'],
                    'OT0571': ['A16,A23,A24,A9'],
                    'OT0572': ['B21,B28,B29'],
                    'OT0573': ['A23,A24,A25,B21,B28,B29'],
                    'OT0574': ['B20,B21,B28,B29,B23'],
                    'OT0575': ['A16,B20,B21,B28,B29'],
                    'OT0576': ['B21,B22,B27,B28,B29,A15,A16'],
                    'OT0579': ['B19,B20,B21,B22,B29'],
                    'OT0581': ['A16,A17,A22,A23,A24,B19,B20,B21,B22'],
                    'OT0582': ['B21,B28,B29'],
                    'OT0584': ['A16,A17,A22,A24,B11,B12,B19,B20,B29'],
                    'OT0585': ['A16,A23,A24,A28,B21,B28,B29'],
                    'OT0587': ['A16,B20,B21'],
                    'OT0589': ['A16,A22,A23,A24,B27,B28,B29,B30'],
                    'OT0594': ['A15,A16,B19,B20,B21,B22'],
                    'OT0596': ['A16,A23,A24,B21,B22,B28,B29'],
                    'OT0598': ['A16,A17,A22,A23,B21,B22,B27,B28,A28'],
                    'OT0603': ['A16,A17,A22,A23,A24,B19,B20,B21,B22'],
                    'OT0609': ['A23,A24,A25,B8'],
                    'OT0612': ['A16,A17,A22,A23,A24'],
                    'OT0613': ['A15,A16,A17,A23,A24'],
                    'OT0618': ['B19,B21,B22,B27,B32'],
                    'OT0619': ['B21,B22,B27,B28'],
                    'OT0621': ['B21,B22,B26,B27,B28'],
                    'OT0623': ['A16,A17,A22,A23,B19,B20,B21,B22'],
                    'OT0624': ['A16,A17,A22,A23'],
                    'OT0629': ['B26,B27,B28,B29,B30'],
                    'OT0632': ['A16,A17,A22,A23,A24'],
                    'OT0633': ['A15,A17,A22,A23'],
                    'OT0634': ['A15,A16,B21,B22,B27,B28,B29'],
                    'OT0636': ['B21,B27,B29,B30'],
                    'OT0637': ['A15,A16,A17,A18'],
                    'OT0638': ['B28,B29,B30'],
                    'OT0639': ['A16,A23,A24,B21,B22,B28'],
                    'OT0641': ['B21,B22,B27,B28,B29'],
                    'OT0642': ['B19,B20,B21,B22,B23'],
                    'OT0643': ['A15,B19,B20,B21,B22'],
                    'OT0644': ['A15,B19,B20,B21,B22'],
                    'OT0646': ['B28,B29,B30'],
                    'OT0648': ['B21,B28,B29'],
                    'OT0650': ['A14,A16,A17,B11,B19,B20,B21'],
                    'OT0651': ['A15,A16,A17,A23,B19,B20,B21,B22'],
                    'OT0652': ['A16,A22,A23,A24,A25,B17'],
                    'OT0653': ['B20,B21,B22,B27,B28,A10,B32,A27'],
                    'OT0654': ['A16,A17,A23,A24,B29'],
                    'OT0656': ['A15,A16,A17,A22,A23'],
                    'OT0657': ['B21,B29,B30'],
                    'OT0659': ['A16,A17,A23,A24'],
                    'OT0660': ['A14,A16,A17,A18,A22'],
                    'OT0663': ['B19,B20,B21,B22,B27'],
                    'OT0665': ['B21,B22,B28,B29'],
                    'OT0666': ['A17,A18,A22,A23'],
                    'OT0668': ['B19,B20,B21,B22,B23,B26,B27,B28,B29,B30'],
                    'OT0670': ['B19,B20,B21,B22,B29,A7'],
                    'OT0671': ['A15,A16,A17,A22,A23,A24'],
                    'OT0673': ['A16,A17,A22,A23,A24,B21,B22,B27,B28,B29'],
                    'OT0675': ['B22,B28,B29'],
                    'OT0676': ['A15,A16,A17,A22,A23,A24,B21,B22,B27,B28,B29'],
                    'OT0678': ['A14,A15,A16,A17,A22,A23,A24,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13'],
                    'OT0679': ['B19,B20,B21,B22,B29'],
                    'OT0683': ['B11,B19,B20,B21,B22'],
                    'OT0684': ['B20,B21,B28,B29,B25,B30'],
                    'OT0685': ['A15,A16,A17,A22,A23,A24,A25'],
                    'OT0686': ['B21,B22,B26,B27,B28,B29,B30'],
                    'OT0687': ['A15,A16,B20,B21,B22,B24,B27,B28,B29'],
                    'OT0688': ['A15,A16,A17,A18,A21,A22,A23'],
                    'OT0689': ['B20,B21,B22,B26,B27,B28'],
                    'OT0690': ['B21,B28,B29'],
                    'OT0691': ['A16,A17,A22,A23,A24,A25'],
                    'OT0693': ['A16,A22,A23,A24,A25'],
                    'OT0695': ['B20,B21,B27,B28,B29,B30'],
                    'OT0696': ['B21,B27,B28,B29,B30'],
                    'OT0698': ['A15,A16,A17,A22,A23,A24'],
                    'OT0699': ['A15,A16,A17,A21,A22,A23,A24,A25'],
                    'OT0701': ['A15,A16,A17,A22,A23,A24,B20,B21'],
                    'OT0702': ['A15,A16,A17,A22,A23,A24,B11,B12,B19,B20,B29'],
                    'OT0703': ['A13,A14,A17,A18,A21,A22,B11,B20'],
                    'OT0704': ['A16,A17,A23,A24,A25'],
                    'OT0705': ['B20,B21,B22,B27,B28,B29'],
                    'OT0706': ['A15,A16,A17,A22,A23,A24,B11,B19,B20,B21,B22,B29'],
                    'OT0707': ['A15,A16,A17,A22,A23,A24'],
                    'OT0708': ['B20,B21,B22,B27,B28,B29'],
                    'OT0709': ['A15,A16,A17,A22,A23,A24'],
                    'OT0710': ['A15,A16,A17,A22,A24,A25,B20,B21,B22,B27,B28,B29'],
                    'OT0711': ['A15,A16,A20,B21,B22,B23,B26,B27,B28,B29'],
                    'OT0717': ['A15,A16,A17,A22,A23,A24,A25,B19,B20,B21,B22,B27,B28,B29'],
                    'OT0718': ['A14,A15,A16,A17,A22,A23,A24,B20,B21,B22,B27,B28,B29'],
                    'OT0723': ['A16,A17,B11,B12,B18,B20,B21,B22,A30'],
                    'OT0725': ['B21,B28,B29,A23'],
                    'OT0730': ['A15,A16,A17,A22,A23,A24,B20,B21'],
                    'OT0732': ['A15,B18,B19,B20,B21,B22,B23'],
                    'OT0733': ['A15,B19,B20,B21,B22'],
                    'OT0734': ['A14,A15,A16,A17,A22,A23,A24,B19,B20,B21,B22'],
                    'OT0736': ['A14,A15,A16,A17,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11'],
                    'OT0737': ['B19,B20,B21,B22,B31'],
                    'OT0742': ['A15,A16,A17,A22,A23'],
                    'OT0744': ['A15,B20,B21,B22,B28,B27,B29'],
                    'OT0745': ['A15,A16,A17,A22,A23,B19,B20,B21,B22,B29'],
                    'OT0748': ['A15,A16,A17,A22,A24,B20,B21,A2,B3'],
                    'OT0752': ['A15,A16,A17,A21,A22,A23,B8,B25'],
                    'OT0753': ['A15,A16,A22,A23,A24,A17'],
                    'OT0755': ['B20,B21,B27,B29,B17,B18,B19,B23,B25'],
                    'OT0756': ['A14,A15,A16,A17'],
                    'OT0757': ['A14,A15,A16,A17,A21,A22,A23,A24'],
                    'OT0759': ['A15,A16,B19,B20,B21,B22,B27,B28'],
                    'OT0761': ['A16,A17,A22,A23,A24'],
                    'OT0762': ['B20,B21,B22,B27,B28,B29,B5,B11,B31'],
                    'OT0763': ['A15,A16,A17,A22,A23,A24'],
                    'OT0766': ['B21,B22,B27,B28'],
                    'OT0767': ['A14,A15,A16,A17,A23,B19,B20,B21,B22,B27,B28'],
                    'OT0771': ['B19,B20,B21,B22,B27,B28,B29,B23'],
                    'OT0773': ['A16,A22,A23,A24,B21,B22,B27,B28,B29,A30,B25,B31'],
                    'OT0774': ['B21,B22,B27,B28,B29'],
                    'OT0776': ['A16,A17,A22,A23,A24,A30,A31'],
                    'OT0778': ['A14,A15,A16,A17,A22,A23,A24,B1'],
                    'OT0779': ['B25,A18,B25'],
                    'OT0781': ['A15,A16,A17,A22,A23,A24'],
                    'OT0782': ['A16,A17,A23,A24,B25'],
                    'OT0783': ['A15,A16,A17,A21,A22,A23,A24,B25'],
                    'OT0784': ['B19,B20,B21,B22,B28,B25'],
                    'OT0786': ['B20,B21,B22,B28,B29,A7,B3'],
                    'OT0788': ['A15,A16,B20,B21,B27,B28,B29,B30'],
                    'OT0791': ['A14,A15,A16,A17,A22,A23,A24,B11,B12,B19,B20,B21,B29'],
                    'OT0792': ['A15,B20,B21,B22,B27,B28,B29,A2'],
                    'OT0793': ['B20,B21,B22,B27,B28,B29'],
                    'OT0795': ['B20,B21,B22,B27,B28,B29,A30'],
                    'OT0796': ['A16,A17,A22,A23,A24,B25,B31'],
                    'OT0797': ['B19,B20,B21,B22,B23,B27,B28,B25'],
                    'OT0802': ['A16,A23,A24,B19,B20,B21,B23,B27'],
                    'OT0803': ['A16,A17,A21,A24,A25,A15'],
                    'OT0805': ['A16,A17,A22,A23,A24,B20,B21,B22,B27,B28,B29'],
                    'OT0806': ['A13,A14,A15,A16,A17,A18,B25'],
                    'OT0807': ['A14,A15,A16,A17,A22,A23,A24,B18,B19,B20,B21,B22'],
                    'OT0808': ['A16,A17,A15,A23,A24,A9,A10,A11,A12,A13,A14,B25'],
                    'OT0810': ['A24,B11,B12,B19,B20,B21,B22'],
                    'OT0816': ['B21,B22,B27,B28,B29,B30'],
                    'OT0817': ['B21,B22,B27,B28,B29,B30,B16,B17,B18,B19,B20,B23,B24,B25,B26'],
                    'OT0818': ['A16,A17,A21,A22,A23,A24,A25'],
                    'OT0820': ['B20,B21,B22,B27,B28,B29,B30'],
                    'OT0822': ['B19,B20,B21,B22,B27,B28,B29'],
                    'OT0824': ['A15,A16,A17,A22,A23,A24,B19,B20,B21,B22,B28,B29'],
                    'OT0826': ['A14,A15,A16,A17,A18,A22,A23,A24,A4'],
                    'OT0830': ['A14,A15,A16,A17,A22,A23,A24,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13'],
                    'OT0832': ['A14,A15,A16,A17,A22,A23,A24,B25'],
                    'OT0836': ['A15,A16,A17,A22,A23,A24,B25'],
                    'OT0847': ['A15,A16,A17,A22,A23,A24'],
                    'OT0848': ['A14,A15,A16,A17,A18,A21,A22,A23,A24,B20,B21'],
                    'OT0851': ['A15,A16,A23,A24,B19,B20,B21,B22'],
                    'OT0854': ['A14,A15,A16,A17,A22,A23,A24,B20,B21,B22,B23,B27,B28'],
                    'OT0855': ['B20,B21,B22,B27,B28,B29,B31'],
                    'OT0859': ['A15,A17,A22,B19,B21,B22,B29'],
                    'OT0861': ['A15,A16,A17'],
                    # 'OT0865': ['A16,A22,A23,B20,B21,A17'], # Odd trigger values - skip for now
                    # 'OT0868': ['A14,A15,A16,A17,A18,A22,A23,A24,B20,B21,B22,B23,B26,B27,B28,B29,B24,B25'],
                    }
# NOTE ABOUT ABOVE: Ask Inyong about why OT0865 and OT0868 seem to have a 
# trigger scheme that is 180 + (MATLAB VALUE*256) as opposed to the previous
# two patterns defined above

# For ease of importing from HANG data spreadsheet, above dictionary was created
# by removing spaces from log entries and using a formula to create the dictionary.
# To preserve code below, iterate over dictionary and split entry by ',' to create
# a list with a separate item for each channel removed.
for subject in removed_channels:
    removed_channels[subject] = removed_channels[subject][0].split(',')
    
# To preserve my own sanity, make sID list from dictionary keys
# NOTE: Several of these sIDs have folders with additional characters at the end
sIDs = [subject for subject in removed_channels]

# Check which sIDs already have -epo.fif files in epochsFolder
if reprocess_data:
    already_processed = []
else:
    already_processed = [file[0:6] for file in os.listdir(epochsFolder) if file.endswith('-epo.fif')]

for sID in sIDs:
    if sID in already_processed:
        continue
    else:
        # New method to match a few wierd folder names
        directory = [folder for folder in os.listdir() if folder.startswith(sID)][0]
        for file in os.listdir(directory):
            if file.endswith(".mat"):
                data_name = os.path.join(directory, file)
                data = loadmat(data_name)
    
        # data['list'] is a 120 x 4 cell array storing 1-item "lists" in each cell
        # Each row is an item set from ITCP
        # For the ISNT experiment script, the target word is always the first column
        # in each row - response order is randomized per trial within the experiment
        # script
        item_sets = data['list']
        
        # data['twOrderP'] and data['twOrderNP'] are both 1 x 121 arrays that
        # reference the row to reference in data['list'] for each trial of a particular
        # condition    
        high_SNR_order = list(data['twOrderP'][0])
        low_SNR_order = list(data['twOrderNP'][0])
        
        # data['isPrimedSeq'] is a 1 x 242 array that references (high vs low SNR) for
        # each trial (e.g. which condition order set to check to lookup correct row in
        # item set list)
        condition_order = list(data['isPrimedSeq'][0])
        
        target_words = []
        counter_high = 0
        counter_low = 0
        
        for trial in condition_order:
            if trial == 1:
                condition = 'HighSNR'
                # Subtract 1 from the high_SNR_order report due to Python / Matlab indices
                target_row = high_SNR_order[counter_high] - 1
                counter_high += 1
                # The first [0] references a 1x1 array, the second [0] gets the
                # contents of the only cell in that array (the string of the target)
                target = str(item_sets[target_row][0][0])
                target_words.append(target)
            elif trial == 0:
                condition = 'LowSNR'
                # Subtract 1 from the low_SNR_order report due to Python / Matlab indices
                target_row = low_SNR_order[counter_low] - 1
                counter_low += 1
                # The first [0] references a 1x1 array, the second [0] gets the
                # contents of the only cell in that array (the string of the target)
                target = str(item_sets[target_row][0][0])
                target_words.append(target)
            else:
                print('ERROR - condition_order contains unexpected value (not 0 or 1)')
                break
            
        for file in os.listdir(directory):
            if file.endswith('.bdf'):
                raw_name = os.path.join(directory, file)
        
        raw = mne.io.read_raw_bdf(raw_name)
        
        if len(raw.info.ch_names) == 73:
            raw.set_channel_types(mapping=
                                  {'EXG1': 'eog',
                                   'EXG2': 'eog',
                                   'EXG3': 'eog',
                                   'EXG4': 'eog',
                                   'EXG5': 'eog',
                                   'EXG6': 'eog',
                                   'EXG7': 'eog',
                                   'EXG8': 'eog'})
        
        raw.set_montage(montage)
        raw.load_data()
        
        # Now using a custom version of BPF.m that is used in MATLAB
        # to make a custom filter with much shorter length to account for
        # potential ringing in the time domain of the CI artifact
        # data[:-1,:] gets all but the last channel for filtering (last channel is stimulus channel)
        # raw._data[:-1,:] = BPF(data=raw._data[:-1,:], fs=2048, norder=256, cf1=1, cf2=57)
        
        # Reverting to using MNE Python's filter defaults to create an epochs
        # object suitable for ICA
        raw.filter(2,45)
        
        raw_events = mne.find_events(raw, shortest_event=1)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
        if min(raw_events[:,2]) > 64000:
            metadata, events, event_id = mne.epochs.make_metadata(events=raw_events, event_id=alt_event_dict,
                                                                  tmin=-0.5, tmax=10.0, sfreq=raw.info['sfreq'],
                                                                  row_events = ['female/HighSNR', 'female/LowSNR',
                                                                                'male/HighSNR', 'male/LowSNR'],
                                                                  keep_first = 'response')
        else:
            metadata, events, event_id = mne.epochs.make_metadata(events=raw_events, event_id=event_dict,
                                                                  tmin=-0.5, tmax=10.0, sfreq=raw.info['sfreq'],
                                                                  row_events = ['female/HighSNR', 'female/LowSNR',
                                                                                'male/HighSNR', 'male/LowSNR'],
                                                                  keep_first = 'response')
        epochs = mne.Epochs(raw=raw,events=events,event_id=event_id, metadata=metadata,
                            tmin=-0.5,tmax=2.1,baseline=None,picks=picks,preload=True)
        # Downsample to 512 Hz
        epochs.resample(512)
        # Add removed channels information from subject log
        epochs.info['bads'] = removed_channels[sID]
        # Add target word order to metadata
        if len(epochs) == len(target_words):
            epochs.metadata['TargetWord'] = target_words
        else:
            print("Length of target_words (from .mat file) does not equal length of epochs for " +sID)
            print("Metadata has not been updated to include target_words - recheck manually")
        fname = os.path.join(cwd, epochsFolder, sID + '-epo.fif')
        epochs.save(fname, overwrite=reprocess_data)