import wfdb
import numpy as np
import os
import re
import glob

os.chdir('./Apnea_data')

physionet_folder = 'physionet_apnea'
export_folder = 'exports'


def merge_data_w_ann(subject: str):

    # Load record and annotations
    record = wfdb.rdsamp(subject)
    ann_record = wfdb.rdann(subject, extension="apn")

    # Extract ECG data and apnea annotations
    ecg_data = record[0].transpose()
    annotations = np.array(ann_record.symbol)

    # Extract other relevant info
    fs = record[1]['fs']
    sig_length = record[1]['sig_len']

    # Determine the number of minutes overlap between ecg data and samples
    min_ann = annotations.shape[0]
    min_data = int(np.floor(sig_length / (fs * 60)))
    minutes = min([min_ann, min_data])

    # Loop to save ECG data combined with annotations in a numpy array
    ecg_w_ann = np.empty((minutes, 60 * fs + 1))  # create empty np array to save ECG data combined with annotations
    # with the first column containing the annotations and the 2 till last column the ecg data. Every row
    # corresponds to one minute
    for i in range(0, minutes):
        if annotations[i] == 'A':
            label = 1
        elif annotations[i] == 'N':
            label = 0
        ecg_w_ann[i, 0] = label
        ecg_w_ann[i, 1:] = ecg_data[0, i * fs * 60:(i + 1) * fs * 60]

    return ecg_w_ann


# sorted list containing subject paths
subjects = sorted([f"./{s.split('.')[1]}" for s in glob.glob(f'./{physionet_folder}/*.hea') if re.search(r'[abc][0-9]+\.hea$', s)])

# Loop through subjects to merge and save to csv per subject and merge all data into one file with all minutes
ecg_w_ann_all = np.empty((0,6001))
for subject in subjects:
    ecg_w_ann = merge_data_w_ann(subject)
    np.savetxt(f"./{export_folder}/{name}.csv", ecg_w_ann, delimiter=",")
    ecg_w_ann_all = np.concatenate((ecg_w_ann_all,ecg_w_ann))

# Save merge of all data into one big file with all minutes of all patients
np.savetxt(f"./{export_folder}/all_subjects.csv", ecg_w_ann_all, delimiter=",")

