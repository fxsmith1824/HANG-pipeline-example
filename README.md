# HANG-pipeline-example
 Example python code for EEG preprocessing pipeline

 A set of numbered python scripts to carry out the current HANG preprocessing pipeline for EEG data. Raw BDF files are too large to be stored in standard GitHub repositories. I have included one epoched data file (located in "1_epochs_w_excluded_channel_info") for demonstration purposes. This epoched file is the output from the python script "1-Epoching.py" - raw BDF files are available on the HANG RDSS drive.
 
 The "mne-package-list.txt" file can be used to create a miniconda/anaconda environment that is identical to mine to eliminate concerns about package version differences.
 
 Some of these scripts may need adjustments to file path information - I have not tested these versions when running anywhere other than our RDSS drive.