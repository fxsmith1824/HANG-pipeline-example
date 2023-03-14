# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 02:07:50 2022

Every ICA fit will throw a warning because of our manual BPF function - the
epochs object is not marked as filtered because we did not use MNE Python 
filter function. It is fine to ignore this, but look into fully implementing
our filter function within MNE so that epochs.info is correctly marked with
bandpass filter parameters.

After speaking with Inyong on 10/13/2022 we decided that because of our weak
highpass filtering (due to shorter filter length) the data need to be baselined
before ICA is applied. If not, the resulting N components will almost
exclusively reflect CI components at various epochs and not neural activity.

In future processing pipelines, this baseline could be applied at step 1 
(during initial epoch object creation, after raw data have been filtered) which
would save computational time during the histogram rejection phase. I am simply
skipping that step for now because I don't want to reprocess 172 ISNT files.

Reminder that after ICA we may want to baseline again to correct for any DC 
shifts.

@author: Francis
"""

import mne
import os

#######################################
# If reprocess_data is True, change file saving overwring to be True
# And skip the check for already created -epo.fif files for each sID
reprocess_data = False

cwd = os.getcwd()
ICA_folder = '2_ICA_set'

sIDs = [subject[:6] for subject in os.listdir(ICA_folder) if subject.endswith('-epo.fif')]

# Check which sIDs already have -epo.fif files in epochsFolder
if reprocess_data:
    already_processed = []
else:
    already_processed = [file[0:6] for file in os.listdir(ICA_folder) if file.endswith('-ica.fif')]

for sID in sIDs:
    if 'noEEG' in sID:
        continue
    elif sID in already_processed:
        continue
    else:
        fname = sID + '-epo.fif'
        path = os.path.join(cwd, ICA_folder, fname)
        # No longer need to apply baseline as data have a stronger highpass filter
        # See 1-Epoching for details
        epochs = mne.read_epochs(path, preload=False)
        # epochs.apply_baseline((-0.2,0))
        epochs.load_data()
        
        ica = mne.preprocessing.ICA(random_state=97, max_iter=800)
        ica.fit(epochs)
        
        ica_fname = os.path.join(cwd, ICA_folder, sID + '-ica.fif')
        ica.save(ica_fname, overwrite=reprocess_data)