# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:38:30 2022

Applies CIAC algorithm to automatically identify potential CI artifacts

@author: Francis
"""

import mne
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def CIAC(epochs, ica, path_bem, path_trans, auditory_onset = 0.0, auditory_offset = None,
         aep_window = (0.080, 0.250), rv_thresh = 20.0, ratio_thresh = 1.5, 
         corr_thresh = 0.9, joint_ratio_thresh = 1.2, joint_corr_thresh = 0.4,
         ratio_extreme = 5.0):
    '''
    A Python implementation of the CIAC algorithm as described in 
    https://doi.org/10.1016/j.heares.2011.12.010
    
    After some initial testing with lab-internal data I have decided to add a 
    final couple of steps to identify additional CI component candidates.
    
    As in CIAC, the first step is to identify CI artifact components based on
    the residual variance of dipoles fitted for each ICA component.
    
    A CI artifact topography template is computed based on the component (among
    those selected in the previous step) with the highest ratio of the 
    component's root-mean-square first derivative during the 50ms following the
    onset and offset of auditory stimuli compared to that component's 
    root-mean-square first derivative during the AEP window. This component is
    also marked as a CI artifact.
    
    All components above the residual variance threshold are evaluated. If they
    EITHER have a RMS ratio greater than ratio_thresh OR have a correlation
    to the template topography greater than corr_thresh they are marked as a CI
    artifact.
    
    My additional steps (based on internal testing) is to then do a final pass
    of all components not already marked as CI artifacts. If these components
    have BOTH a ratio greater than joint_ratio_thresh AND a correlation to the
    template topography greatern joint_corr_thresh they are marked as a CI 
    artifact.
    
    In addition, if any component (regardless of residual variance) has a ratio
    greater than ratio_extreme it is also marked as a CI artifact.
    
    These final two steps probably need more extensive testing. Also note that
    in the current HANG pipeline, the ICAs being used will still need to be
    manually reviewed for eyeblinks, eye movements, etc.
    
    NOTE: Should probably save dipole fit so it doesn't need to be re-run if
    data are reviewed.

    Parameters
    ----------
    epochs : Instance of mne.Epochs class object
        The epochs object for which CI artifacts should be removed/corrected.
    ica : Instance of mne.preprocessing.ICA class object
        The ICA which has been fit in a previous processing step to the epoched
        data - in order to identify components which reflect the CI artifact.
    path_bem : STRING
        Path to the BEM files needed for dipole fitting.
    path_trans : STRING
        Path to the head <-> MRI transform file.
    auditory_onset : FLOAT, optional
        The time (relative to the epoched stimuli) of the auditory stimulus 
        onset. The default is 0.0.
    auditory_offset : FLOAT, optional
        If the auditory stimuli in the current experiment are of uniform 
        duration, this should reflect the time (relative to the epoched
        stimuli) of the auditory stimulus offset. If your stimuli are of 
        non-uniform length, use the default value of None. The default is None.
    aep_window : TUPLE of FLOAT, optional
        The time window during which the Auditory Evoked Potential (N1/P2) is
        expected to occur. The default is (0.080, 0.250).
    rv_thresh : FLOAT, optional
        The threshold for residual variance of ICA component dipole fits to 
        consder for CI artifacts. Dipoles with low residual variance are more
        likely to represent neural components Anything over this threshold is
        included in the initial pass considering potential CI artifacts. This 
        value should not be changed within a given study-set of participants.
        The default is 20.
    ratio_thresh : FLOAT, optional
        The threshold for considering an ICA component as a potential CI 
        artifact. This is the ratio of the RMS of the first derivative of a 
        given ICA component during the 50ms after auditory onset + offset 
        compared to the first derivative of that component during the AEP 
        window. The default is 1.5.
    corr_thresh : FLOAT, optional
        The threshold for considering an ICA component as a potential CI 
        artifact. This is the correlation between the topography of a given
        ICA component and the "template topography for a given participant. The
        default is 0.9.
    joint_ratio_thresh : FLOAT, optional
        A lower threshold for considering ICA components in combination with
        joint_corr_thresh. This is intended to catch components which are 
        likely CI artifacts but are not caught individually by either ratio or 
        correlation. The default is 1.2.
    joint_corr_thresh : FLOAT, optional
        A lower threshold for considering ICA components in combination with
        joint_ratio_thresh. This is intended to catch components which are 
        likely CI artifacts but are not caught individually by either ratio or
        correlation. The default is 0.4.
    ratio_extreme : FLOAT, optional
        A higher ratio threshold for considering ICA components which, based on
        residual variance, was not considered on the first pass but still 
        reflects much more activity during the CI artifact window than during
        the AEP window. The default is 5.0.

    Returns
    -------
    None.
    
    This modifies the ICA in place. You will still need to save the modified
    ICA object.

    '''
    # Get ICA sources for estimating CI artifact and N1 derivatives
    sources = ica.get_sources(inst=epochs)
    # The average (evoked-ish) of the ICA scources are the data for the AU timecourse plots for each component
    source_avg = sources.average(picks=sources.info['ch_names'])
    df = source_avg.to_data_frame()
    # Set up dataframes for CI artifact window and AEP window
    df_ci_on = df.loc[(df['time'] >= auditory_onset) & (df['time'] <= auditory_onset+0.050)]
    if auditory_offset:
        df_ci_off = df.loc[(df['time'] >= auditory_offset) & (df['time'] <= auditory_offset +0.050)]
        df_ci = pd.concat([df_ci_on, df_ci_off])
    else:
        df_ci = df_ci_on
    df_n1 = df.loc[(df['time'] >= aep_window[0]) & (df['time'] <= aep_window[1])]
    # Compute RMS ratio for first derivative of components during CI artifact 
    # window and AEP window
    df_rms = pd.DataFrame(columns=['component', 'ci_rms', 'n1_rms', 'ratio', 'int_component'])
    for component in source_avg.info['ch_names']:
        ci_rms = np.sqrt(np.mean(np.gradient(df_ci[component])**2))
        n1_rms = np.sqrt(np.mean(np.gradient(df_n1[component])**2))
        ratio = ci_rms / n1_rms
        int_component = int(component[3:])
        df_rms.loc[len(df_rms)] = [component, ci_rms, n1_rms, ratio, int_component]
    # Get topographies for all components as dataframe
    df_topo = pd.DataFrame(data=ica.get_components(), columns = source_avg.info['ch_names'])
    topo_corr = abs(df_topo.corr())
    # Fit dipoles to each component (this step takes a while)
    noise_cov = mne.compute_covariance(epochs, tmin=-0.4, tmax=-0.2)
    components = mne.EvokedArray(df_topo, ica.info, tmin=0.0, nave=len(epochs))
    components.set_eeg_reference()
    dipole, res = mne.fit_dipole(components, noise_cov, path_bem, trans=path_trans)
    df_residuals = pd.DataFrame(columns=['component', 'GOF', 'residual', 'int_component'])
    for component in source_avg.info['ch_names']:
        int_component = int(component[3:])
        gof = dipole.gof[int_component]
        rv = 100-gof
        result = [component, gof, rv, int_component]
        df_residuals.loc[len(df_residuals)] = result
    # Now apply CIAC criteria to flag components
    options_threshold = []
    to_exclude = []

    # First pass - choose all components with residual variance > threshold
    for component in source_avg.info['ch_names']:
        if df_residuals.loc[df_residuals['component']==component]['residual'].values[0] > rv_thresh:
            options_threshold.append(component)

    # Subset df_rms based on rv_thresh qualifications, sort, and pick template
    df_rms_thresh = df_rms[df_rms['component'].isin(options_threshold)]
    df_rms_thresh = df_rms_thresh.sort_values(by='ratio', ascending=False)
    template = df_rms_thresh['component'].iloc[0]
    to_exclude.append(template)

    # Now iterate over all the options not already in to_exclude and see if they
    # belong in to_exclude
    remaining_options = [x for x in options_threshold if x not in to_exclude]
    for component in remaining_options:
        if df_rms[df_rms['component'] == component]['ratio'].values[0] > ratio_thresh:
            to_exclude.append(component)
        elif topo_corr[template][component] > corr_thresh:
            to_exclude.append(component)
        else:
            continue

    all_options = [x for x in source_avg.info['ch_names'] if x not in to_exclude]
    for component in all_options:
        if (df_rms[df_rms['component'] == component]['ratio'].values[0] > joint_ratio_thresh
            and topo_corr[template][component] > joint_corr_thresh):
            to_exclude.append(component)

    extreme_cases = list(df_rms[df_rms['ratio'] > ratio_extreme]['component'].values)
    extreme_cases = [x for x in extreme_cases if x not in to_exclude]
    to_exclude += extreme_cases

    # Convert to exclude to integers
    to_exclude = [int(x[3:]) for x in to_exclude]
    ica.exclude = to_exclude
    return dipole

#######################################
# If reprocess_data is True, change file saving overwring to be True
# And skip the check for already created -epo.fif files for each sID
reprocess_data = False

cwd = os.getcwd()
ICA_folder = '2_ICA_set'

SUBJECTS_DIR='\\\\iowa.uiowa.edu\\shared\\ResearchData\\rdss_inychoi\\StructuralMRIdata\\'
SUBJECT = 'fsaverage'
path_label = SUBJECTS_DIR + SUBJECT + '/label/'
path_bem = SUBJECTS_DIR + SUBJECT + '/bem/'
path_bem += 'fsaverage-5120-5120-5120-bem-sol.fif'
path_trans = os.path.join(cwd, 'Biosemi64median206subjects10percentLarger-trans.fif')

# Get list of all subjects who have had ICA run
all_sIDs = list(set([subject[:6] for subject in os.listdir(ICA_folder) if subject.endswith('-ica.fif')]))
already_processed = [subject[:6] for subject in os.listdir(ICA_folder) if subject.endswith('CIAC-ica.fif')]

if reprocess_data == True:
    sIDs = all_sIDs
else:
    sIDs = [x for x in all_sIDs if x not in already_processed]

for sID in sIDs:
    fname = sID + '-epo.fif'
    epochs = mne.read_epochs(os.path.join(cwd, ICA_folder, fname))
    fname_ica = sID + '-ica.fif'
    ica = mne.preprocessing.read_ica(os.path.join(cwd, ICA_folder, fname_ica))
    dipole = CIAC(epochs, ica, path_bem, path_trans, auditory_offset=2.0)
    ica_fname = os.path.join(cwd, ICA_folder, sID + '-CIAC-ica.fif')
    ica.save(ica_fname, overwrite=reprocess_data)
    dipole_fname = os.path.join(cwd, ICA_folder, sID + '-CIAC.dip')
    dipole.save(dipole_fname, overwrite=reprocess_data)
