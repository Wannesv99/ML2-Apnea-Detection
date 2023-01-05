import wfdb
import numpy as np
import pandas as pd

physionet_folder = 'physionet_apnea'


def merge_data_w_ann(subject):
    # filename has to be in string format

    # Define record path
    record_path = "./" + physionet_folder + "/" + subject

    # Load record and annotations
    record = wfdb.rdsamp(record_path)
    ann_record = wfdb.rdann(record_path, extension="apn")

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
        ecg_w_ann[i, 1:] = ecg_data[0][i * fs * 60:(i + 1) * fs * 60]

    return ecg_w_ann


# Loop through data files of subjects, merge ecg data and annotations to np array and change into csv format
import os
import re

# Get subject names into subject list
files = os.listdir("./"+physionet_folder)
subjects = []
for file in files:
    parts_file = re.split('\.', file)
    parts_file[0]
    if (len(parts_file)>1) and (len(parts_file[0])==3):
        if parts_file[1] == 'apn':
            subjects.append(parts_file[0])


# Loop through subjects to merge and save to csv per subject
for subject in subjects:
    ecg_w_ann = merge_data_w_ann(subject)
    df = pd.DataFrame(ecg_w_ann)
    df.to_csv("./" + physionet_folder + "/" + subject + ".csv", header=False, index=False)
    #np.savetxt("./"+physionet_folder+"/"+subject+".csv", ecg_w_ann, delimiter=",")