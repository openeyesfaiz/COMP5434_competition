import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 600
from tqdm import tqdm
import gc

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Dense, Activation, Dropout, Flatten

from keras import regularizers 

import keras

from sklearn.model_selection import GroupKFold

def init():
    np.random.seed = 0
    
init()


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape2D(y_true, y_pred):
    return smape(np.ravel(y_true), np.ravel(y_pred))
    
def smape_mask(y_true, y_pred, threshold):
    denominator = (np.abs(y_true) + np.abs(y_pred)) 
    diff = np.abs(y_true - y_pred) 
    diff[denominator == 0] = 0.0
    
    return diff <= (threshold / 2.0) * denominator

max_size = 181 

offset = 1/2

train_all = pd.read_csv("data/train_2.csv")
train_all.head()

all_page = train_all.Page.copy()
train_key = train_all[['Page']].copy()
train_all = train_all.iloc[:,1:] * offset 
train_all.head()

def get_date_index(date, train_all=train_all):
    for idx, c in enumerate(train_all.columns):
        if date == c:
            break
    if idx == len(train_all.columns):
        return None
    return idx

get_date_index('2017-09-10') - get_date_index('2016-09-10')

trains = []
tests = []

train_end = get_date_index('2016-09-10') + 1
test_start = get_date_index('2016-09-13')

for i in range(-3,4):
    train = train_all.iloc[ : , (train_end - max_size + i) : train_end + i].copy().astype('float32')
    test = train_all.iloc[:, test_start + i : (63 + test_start) + i].copy().astype('float32')
    train = train.iloc[:,::-1].copy().astype('float32')
    trains.append(train)
    tests.append(test)

train = train_all.iloc[ : , (train_end - max_size) : train_end].copy().astype('float32')
test = train_all.iloc[:, test_start : (63 + test_start)].copy().astype('float32')
train = train.iloc[:,::-1].copy().astype('float32')

train_all = train_all.iloc[:,-(max_size):].astype('float32')
train_all = train_all.iloc[:,::-1].copy().astype('float32')

test_3_date = tests[3].columns

train_all.head()

trains[3].head()

trains[0].head()

tests[3].head()

data = [page.split('_') for page in tqdm(train_key.Page)]

access = ['_'.join(page[-2:]) for page in data]

site = [page[-3] for page in data]

page = ['_'.join(page[:-3]) for page in data]
page[:2]

train_key['PageTitle'] = page
train_key['Site'] = site
train_key['AccessAgent'] = access
train_key.head()

train_norms = [np.log1p(train).astype('float32') for train in trains]
train_norms[3].head()

train_all_norm = np.log1p(train_all).astype('float32')
train_all_norm.head()

for i,test in enumerate(tests):
    first_day = i-2 
    test_columns_date = list(test.columns)
    test_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
    test.columns = test_columns_code

tests[3].head()

tests[0].head()

for test in tests:
    test.fillna(0, inplace=True)

    test['Page'] = all_page
    test.sort_values(by='Page', inplace=True)
    test.reset_index(drop=True, inplace=True)
    
tests = [test.merge(train_key, how='left', on='Page', copy=False) for test in tests]

tests[3].head()

tests[0].head()

test_all_id = pd.read_csv('data/key_2.csv')

test_all_id['Date'] = [page[-10:] for page in tqdm(test_all_id.Page)]
test_all_id['Page'] = [page[:-11] for page in tqdm(test_all_id.Page)]
test_all_id.head()

test_all = test_all_id.drop('Id', axis=1)
test_all['Visits_true'] = np.NaN

test_all.Visits_true = test_all.Visits_true * offset
test_all = test_all.pivot(index='Page', columns='Date', values='Visits_true').astype('float32').reset_index()

test_all['2017-11-14'] = np.NaN
test_all.sort_values(by='Page', inplace=True)
test_all.reset_index(drop=True, inplace=True)

test_all.head()

print(test_all.shape)


test_all_columns_date = list(test_all.columns[1:])
first_day = 2 
test_all_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
cols = ['Page']
cols.extend(test_all_columns_code)
test_all.columns = cols
test_all.head()

test_all = test_all.merge(train_key, how='left', on='Page')
print(test_all.head())


y_cols = test.columns[:63]
print(y_cols)

for test in tests:
    test.reset_index(inplace=True)
test_all = test_all.reset_index()

print(test_all.shape)

test = pd.concat(tests[2:5], axis=0).reset_index(drop=True)
print(test.shape)

print(test.head())
print(test.tail())

test_all = test_all[test.columns].copy()
test_all.head()

train_cols = ['d_%d' % i for i in range(train_norms[0].shape[1])]
print(len(train_cols))

for train_norm in train_norms:
    train_norm.columns = train_cols
train_all_norm.columns = train_cols
train_norm = pd.concat(train_norms[2:5], axis=0).reset_index(drop=True)
train_norm.shape

print(train_norm.head())

train_norm_diff = train_norm - train_norm.shift(-1, axis=1)
train_norm_diff.head()

train_all_norm_diff = train_all_norm - train_all_norm.shift(-1, axis=1)
train_all_norm_diff.head()

train_norm_diff7 = train_norm - train_norm.shift(-7, axis=1)
train_norm_diff7.head()

train_all_norm_diff7 = train_all_norm - train_all_norm.shift(-7, axis=1)
train_all_norm_diff7.head()

train_norm = train_norm.iloc[:,::-1]
train_norm_diff7m = train_norm - train_norm.rolling(window=7, axis=1).median()
train_norm = train_norm.iloc[:,::-1]
train_norm_diff7m = train_norm_diff7m.iloc[:,::-1]
train_norm_diff7m.head()


train_all_norm = train_all_norm.iloc[:,::-1]
train_all_norm_diff7m = train_all_norm - train_all_norm.rolling(window=7, axis=1).median()
train_all_norm = train_all_norm.iloc[:,::-1]
train_all_norm_diff7m = train_all_norm_diff7m.iloc[:,::-1]
train_all_norm_diff7m.head()

all(test[:test_all.shape[0]].Page == test_all.Page)

sites = train_key.Site.unique()
print(sites)

test_site = pd.factorize(test.Site)[0]
test['Site_label'] = test_site
test_all['Site_label'] = test_site[:test_all.shape[0]]


accesses = train_key.AccessAgent.unique()
print(accesses)

test_access = pd.factorize(test.AccessAgent)[0]
test['Access_label'] = test_access
test_all['Access_label'] = test_access[:test_all.shape[0]]

print(test.shape)

print(test_all.shape)

test0 = test.copy()
test_all0 = test_all.copy()

y_norm_cols = [c+'_norm' for c in y_cols]
y_pred_cols = [c+'_pred' for c in y_cols]

def add_median(test, train,
               train_key, periods, max_periods, first_train_weekday):
    train =  train.iloc[:,:7*max_periods]
    
    df = train_key[['Page']].copy()
    df['AllVisits'] = train.median(axis=1).fillna(0)
    test = test.merge(df, how='left', on='Page', copy=False)
    test.AllVisits = test.AllVisits.fillna(0).astype('float32')
    
    for site in sites:
        test[site] = (1 * (test.Site == site)).astype('float32')
    
    for access in accesses:
        test[access] = (1 * (test.AccessAgent == access)).astype('float32')

    for (w1, w2) in periods:
        
        df = train_key[['Page']].copy()
        c = 'median_%d_%d' % (w1, w2)
        df[c] = train.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        test = test.merge(df, how='left', on='Page', copy=False)
        test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')

    for c_norm, c in zip(y_norm_cols, y_cols):
        test[c_norm] = (np.log1p(test[c]) - test.AllVisits).astype('float32')

    gc.collect()

    return test

max_periods = 16
periods = [(0,1), (1,2), (2,3), (3,4), 
           (4,5), (5,6), (6,7), (7,8),
           ]


site_cols = list(sites)
access_cols = list(accesses)

test, test_all = test0.copy(), test_all0.copy()

for c in y_pred_cols:
    test[c] = np.NaN
    test_all[c] = np.NaN

test1 = add_median(test, train_norm, 
                   train_key, periods, max_periods, 3)

print("Test1 is:", test1)

test_all1 = add_median(test_all, train_all_norm, 
                       train_key, periods, max_periods, 5)

print("Test_all1 is:", test_all1)

num_cols = (['median_%d_%d' % (w1,w2) for (w1,w2) in periods])

import keras.backend as K

def smape_error(y_true, y_pred):
    return K.mean(K.clip(K.abs(y_pred - y_true),  0.0, 1.0), axis=-1)


def get_model(input_dim, num_sites, num_accesses, output_dim):
    
    dropout = 0.5
    regularizer = 0.00004
    main_input = Input(shape=(input_dim,), dtype='float32', name='main_input')
    site_input = Input(shape=(num_sites,), dtype='float32', name='site_input')
    access_input = Input(shape=(num_accesses,), dtype='float32', name='access_input')
    
    
    x0 = keras.layers.concatenate([main_input, site_input, access_input])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x0)
    x = Dropout(dropout)(x)
    x = keras.layers.concatenate([x0, x])
    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = BatchNormalization(beta_regularizer=regularizers.l2(regularizer),
                           gamma_regularizer=regularizers.l2(regularizer)
                          )(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)

    x = Dense(200, activation='relu', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)
    x = Dropout(dropout)(x)
    x = Dense(output_dim, activation='linear', 
              kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer))(x)

    model =  Model(inputs=[main_input, site_input, access_input], outputs=[x])
    model.compile(loss=smape_error, optimizer='adam')
    return model

group = pd.factorize(test1.Page)[0]

n_bag = 20
kf = GroupKFold(n_bag)
batch_size=4096

test2 = test1
test_all2 = test_all1
X, Xs, Xa, y = test2[num_cols].values, test2[site_cols].values, test2[access_cols].values, test2[y_norm_cols].values
X_all, Xs_all, Xa_all, y_all = test_all2[num_cols].values, test_all2[site_cols].values, test_all2[access_cols].values, test_all2[y_norm_cols].fillna(0).values

y_true = test2[y_cols]
y_all_true = test_all2[y_cols]

models = [get_model(len(num_cols), len(site_cols), len(access_cols), len(y_cols)) for bag in range(n_bag)]

print('offset:', offset)
print('batch size:', batch_size)


best_score = 100
best_all_score = 100

save_pred = 0
saved_pred_all = 0

for n_epoch in range(10, 201, 10):
    print('************** start %d epochs **************************' % n_epoch)

    y_pred0 = np.zeros((y.shape[0], y.shape[1]))
    y_all_pred0 = np.zeros((n_bag, y_all.shape[0], y_all.shape[1]))
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, group)):
        print('train fold', fold, end=' ')    
        model = models[fold]
        X_train, Xs_train, Xa_train, y_train = X[train_idx,:], Xs[train_idx,:], Xa[train_idx,:], y[train_idx,:]
        X_test, Xs_test, Xa_test, y_test = X[test_idx,:], Xs[test_idx,:], Xa[test_idx,:], y[test_idx,:]

        model.fit([ X_train, Xs_train, Xa_train],  y_train, 
                  epochs=10, batch_size=batch_size, verbose=0, shuffle=True, 
                 )
        y_pred = model.predict([ X_test, Xs_test, Xa_test], batch_size=batch_size)
        y_all_pred = model.predict([X_all, Xs_all, Xa_all], batch_size=batch_size)

        y_pred0[test_idx,:] = y_pred
        y_all_pred0[fold,:,:]  = y_all_pred

        y_pred += test2.AllVisits.values[test_idx].reshape((-1,1))
        y_pred = np.expm1(y_pred)
        y_pred[y_pred < 0.5 * offset] = 0
        res = smape2D(test2[y_cols].values[test_idx, :], y_pred)
        y_pred = offset*((y_pred / offset).round())
        res_round = smape2D(test2[y_cols].values[test_idx, :], y_pred)

        y_all_pred += test_all2.AllVisits.values.reshape((-1,1))
        y_all_pred = np.expm1(y_all_pred)
        y_all_pred[y_all_pred < 0.5 * offset] = 0
        res_all = smape2D(test_all2[y_cols], y_all_pred)
        y_all_pred = offset*((y_all_pred / offset).round())
        res_all_round = smape2D(test_all2[y_cols], y_all_pred)
        print('smape train: %0.5f' % res, 'round: %0.5f' % res_round,
              '     smape LB: %0.5f' % res_all, 'round: %0.5f' % res_all_round)

    y_all_pred0  = np.nanmedian(y_all_pred0, axis=0)

    y_pred0  += test2.AllVisits.values.reshape((-1,1))
    y_pred0 = np.expm1(y_pred0)
    y_pred0[y_pred0 < 0.5 * offset] = 0
    res = smape2D(y_true, y_pred0)
    print('smape train: %0.5f' % res, end=' ')
    y_pred0 = offset*((y_pred0 / offset).round())
    res_round = smape2D(y_true, y_pred0)
    print('round: %0.5f' % res_round)

    y_all_pred0 += test_all2.AllVisits.values.reshape((-1,1))
    y_all_pred0 = np.expm1(y_all_pred0)
    y_all_pred0[y_all_pred0 < 0.5 * offset] = 0

    res_all = smape2D(y_all_true, y_all_pred0)
    print('     smape LB: %0.5f' % res_all, end=' ')
    y_all_pred0 = offset*((y_all_pred0 / offset).round())
    res_all_round = smape2D(y_all_true, y_all_pred0)
    print('round: %0.5f' % res_all_round, end=' ')
    if res_round < best_score:
        print('saving')
        best_score = res_round
        best_all_score = res_all_round
        test.loc[:, y_pred_cols] = y_pred0
        test_all.loc[:, y_pred_cols] = y_all_pred0
    else:
        print()
    print('*************** end %d epochs **************************' % n_epoch)
print('best saved LB score:', best_all_score)

filename = 'keras_simple_sept_10'

test_all_columns_save = [c+'_pred' for c in test_all_columns_code]
test_all_columns_save.append('Page')
test_all_save = test_all[test_all_columns_save]

test_all_save.columns = test_all_columns_date+['Page']

test_all_save.to_csv('data/%s_test_all_save.csv' % filename, index=False)

test_all_save_columns = test_all_columns_date[:-1]+['Page']
test_all_save = test_all_save[test_all_save_columns]

test_all_save = pd.melt(test_all_save, id_vars=['Page'], var_name='Date', value_name='Visits')

test_all_sub = test_all_id.merge(test_all_save, how='left', on=['Page','Date'])

test_all_sub.Visits = (test_all_sub.Visits / offset).round()

test_all_sub_sorted = test_all_sub[['Id', 'Visits']].sort_values(by='Id')

test_all_sub_sorted[['Id', 'Visits']].to_csv('submissions/%s_test_sorted.csv' % filename, index=False)

test_all_sub[['Id', 'Visits']].to_csv('submissions/%s_test.csv' % filename, index=False)

filename = 'keras_simple_sept_10'

test_all_columns_save = [c+'_pred' for c in test_all_columns_code]
test_all_columns_save.append('Page')
test_all_save = test_all[test_all_columns_save]

test_all_save.columns = test_all_columns_date+['Page']

test_all_save.to_csv('data/%s_test_all_save.csv' % filename, index=False)

test_all_save_columns = test_all_columns_date[:-1]+['Page']
test_all_save = test_all_save[test_all_save_columns]

test_all_save = pd.melt(test_all_save, id_vars=['Page'], var_name='Date', value_name='Visits')

test_all_sub = test_all_id.merge(test_all_save, how='left', on=['Page','Date'])

test_all_sub.Visits = (test_all_sub.Visits / offset).round()

test_all_sub_sorted = test_all_sub[['Id', 'Visits']].sort_values(by='Id')

test_all_sub_sorted[['Id', 'Visits']].to_csv('submissions/%s_test_sorted.csv' % filename, index=False)

test_all_sub[['Id', 'Visits']].to_csv('submissions/%s_test.csv' % filename, index=False)

test_columns_save = [c+'_pred' for c in test_columns_code]
test_columns_save.append('Page')
test_save = test[test_columns_save]
test_save.shape

test3_save_columns = [c+'_pred' for c in tests[3].columns[1:-4]]
test3_save_columns.append('Page')
test_save = test_save[test3_save_columns].reset_index(drop=True)
test_save.head()

test_save = test_save.iloc[145063:2*145063,:].reset_index(drop=True)

test3_save_columns = list(test_3_date)
test3_save_columns.append('Page')
test_save.columns = test3_save_columns
test_save.head()

test_save = pd.melt(test_save, id_vars=['Page'], var_name='Date', value_name='Visits')
test_save.Visits = (test_save.Visits / offset).round()

test_save.to_csv('submissions/%s_train.csv' % filename, index=False)

test_all_sub.Visits.isnull().sum()

all(test_all.Page == tests[3].Page)


test_save.Visits.isnull().sum()