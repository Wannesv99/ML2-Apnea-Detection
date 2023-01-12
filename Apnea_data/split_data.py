# input of function is dataframe with all minutes of all subjects (17023x6001)
def split_data(concat):
    # create apnea and no_apnea dataframe, based on label in column 0 of concat df
    apnea = concat[concat[0] == 1.0]
    no_apnea = concat[concat[0] == 0.0]

    # drop excess no_apnea data to balance out classes (50/50)
    exclude = no_apnea.shape[0] - apnea.shape[0]
    excl_sample = no_apnea.sample(n=exclude)
    no_apnea.drop(excl_sample.index, inplace=True)

    # take random sample of 80 % of apnea for training, and rest (20%) for test
    train_apnea = apnea.sample(frac=0.8)
    test_apnea = apnea.drop(train_apnea.index)

    # same for no_apnea
    train_no_apnea = no_apnea.sample(frac=0.8)
    test_no_apnea = no_apnea.drop(train_no_apnea.index)

    # concatenate the train and test parts of apnea and no_apnea
    train = pd.concat([train_apnea, train_no_apnea], axis=0)
    test = pd.concat([test_apnea, test_no_apnea], axis=0)

    # shuffle data
    train = train.sample(frac=1)
    test = test.sample(frac=1)

    # rest index
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, test