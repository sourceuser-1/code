import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_reformated_cvs(filename,nrows=25000):
    df = pd.read_csv(filename,nrows=nrows)
    df=df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna(axis=0)
    return df

def create_dataset(dataset, look_back=1, look_forward=1):
    '''
     Description:
         Method use to create a single trace for LSTM model
         This allows for easier data processing within the TF2 Dataset tools
     :param dataset: pandas dataframe with variable
     :param look_back: number of time step before prediction
     :param look_forward: number of time step to prediction
     :return: two numpy array (input,output)
     '''
    X, Y = [], []
    offset = look_back + look_forward
    for i in range(len(dataset) - (offset + 1)):
        xx = dataset[i:(i + look_back), 0]
        yy = dataset[(i + look_back):(i + offset), 0]
        X.append(xx)
        Y.append(yy)
    return np.array(X), np.array(Y)

def get_dataset(dataframe, variable='B:VIMIN', split_fraction=0.8, look_back=15, look_forward=1):
    '''
     Description:
         Method that scales the data and split into train/test datasets
     :param variable: desired variable
     :param dataframe: pandas dataframe
     :param split_fraction: desired split fraction between train and test
     :return: scaler, (x-train,y-train), (x-test,y-test)
    '''
    dataset = dataframe[variable].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * split_fraction)

    ## Split dataset
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    ## Create train dataset
    X_train, Y_train = create_dataset(train, look_back, look_forward)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

    ## Create test dataset
    X_test, Y_test = create_dataset(test, look_back, look_forward)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
    Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))

    return scaler, X_train, Y_train, X_test, Y_test

def get_datasets(dataframe,variables = ['B:VIMIN', 'B:IMINER', 'B:LINFRQ', 'I:IB', 'I:MDAT40'],split_fraction=0.8,concate_axis=2):
    data_list = []
    scalers = []
    for v in range(len(variables)):
        data_list.append(get_dataset(dataframe,variable=variables[v],split_fraction=split_fraction))
    ## TODO: Horrible hack that should quickly be fixed
    scalers = [data_list[0][0],data_list[1][0],data_list[2][0],data_list[3][0],data_list[4][0]]
    X_train = np.concatenate((data_list[0][1], data_list[1][1], data_list[2][1], data_list[3][1], data_list[4][1]), axis=concate_axis)
    Y_train = data_list[0][2]
    X_test = np.concatenate((data_list[0][3], data_list[1][3], data_list[2][3], data_list[3][3], data_list[4][3]), axis=concate_axis)
    Y_test = data_list[0][4]
    return scalers,X_train,Y_train,X_test,Y_test
