
import pandas as pd
import glob

def concat_subjects(files: list) -> pd.DataFrame:
    '''
        Concatenate subject csv files into a single frame

        Params:
        files: list of file paths to be read
    '''
    subject_frames = []
    for f in files:
        df = pd.read_csv(f, header=None)
        subject_frames.append(df)

    res = pd.concat(subject_frames, ignore_index=True).reset_index(drop=True)
    res.rename(columns={ res.columns[0]: 'label' }, inplace=True)
    return res


def train_test_split(subject_df: pd.DataFrame, train_fraction=0.8, train_seed=10):
    '''
        Split frame into train and test sets using the indices of the resampled 
        training set to generate the test set.
        
        Params:
        subject_df: frame containing labelled data.
        train_fraction: relative size of the training set.
        train_seed: constant to fix the random state of the resampling method
    '''
    train_frame = subject_df.sample(frac=train_fraction, random_state=train_seed)
    test_frame  = subject_df.drop(train_frame.index)
    assert len(train_frame) + len(test_frame) == len(subject_df), f"Expected lengths of train and test set to be {len(df)}, got: {len(train) + len(test)}."
    return (train_frame, test_frame)


def balance_classes(df: pd.DataFrame, seed=10) -> pd.DataFrame:
    '''
        Take binary labelled dataframe and resample both classes to match 
        the class with the lowest sample frequency.

        Params:
        subject_df: binary labelled frame to be balanced
        seed: constant to fix the random state of the resampling method
    '''

    c1 = df[df['label'] == 0.]
    c2 = df[df['label'] == 1.]
    
    least_samples = min(len(c1), len(c2))

    # Could be optimized but too lazy
    c1 = c1.sample(n=least_samples, random_state=seed)
    c2 = c2.sample(n=least_samples, random_state=seed)

    assert len(c1) == len(c2), f'Expected classes to have equal n of samples'
    return pd.concat([c1,c2], ignore_index=True)

# list file paths
files = glob.glob('./Apnea_data/exports/*.csv')

# concat all csv's into a single frame and balance apnea and non-apnea segments
df = concat_subjects(files)
df = balance_classes(df)

# split in train and test set 80/20
train, test = train_test_split(df, train_fraction=0.8, train_seed=33)

# Save the balanced train and test data to csv
train.to_csv("./pysionet_apnea/train_balanced.csv", header=False, index=False)
test.to_csv("./pysionet_apnea/test_balanced.csv", header=False, index=False)