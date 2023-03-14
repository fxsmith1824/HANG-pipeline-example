# HANG-pipeline-example
 Example python code for EEG preprocessing pipeline

 A set of numbered python scripts to carry out the current HANG preprocessing pipeline for EEG data. Raw BDF files are too large to be stored in standard GitHub repositories. I have included the output of each step of the scripts in the numbered folders for one participant for demonstration purposes. If you try to run later scripts without setting reprocess_data to True it will give you an error because data files already exist in the output folder. Feel free to delete the contents of the folders 2_ICA_set and 3_mne_epochs_after_rejection if you want to run scripts 2 through 4b for yourself.
 
 The "mne-package-list.txt" file can be used to create a miniconda/anaconda environment that is identical to mine to eliminate concerns about package version differences.
 
 Some of these scripts may need adjustments to file path information - I have not tested these versions when running anywhere other than our RDSS drive.