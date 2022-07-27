def dis_sparse(concat_train,concat_test,one_hot=True):
    from sklearn.preprocessing import OneHotEncoder,LabelEncoder
    lbl = LabelEncoder()
    l = np.vstack((concat_train,concat_test))
    l = np.unique(l)
    lbl.fit(l.reshape(-1,1))
    concat_train= lbl.transform(concat_train.reshape(-1,1))
    concat_test = lbl.transform(concat_test.reshape(-1,1))
    if(one_hot==True):
        one_clf = OneHotEncoder()
        l = np.vstack((concat_train.reshape(-1,1),concat_test.reshape(-1,1)))
        l = np.unique(l)
        one_clf.fit(l.reshape(-1,1))
        sparse_training_matrix= one_clf.transform(concat_train.reshape(-1,1))
        sparse_testing_matrix = one_clf.transform(concat_test.reshape(-1,1))
        return sparse_training_matrix,sparse_testing_matrix
    else:
        return concat_train,concat_test



import pandas as pd
import gc
import os
import time
time_begin = time.time()

dfTrain = pd.read_csv("/data2/ruike/greenplum/cmd/criteo/train_sample.csv")
#dfTrain.iloc[:1000,:].to_csv("/data2/ruike/greenplum/cmd/criteo/train_sample.csv")

dfTest = pd.read_csv("/data2/ruike/greenplum/cmd/criteo/train_sample.csv")



print("load data to RAM: {}".format(time.time() - time_begin))
time_begin = time.time()

global_columns  = dfTrain.columns.tolist()
ID_columns  = ["C"+str(i) for i in range(14, 40)]#["ps_reg_01", "ps_reg_02", "ps_reg_03","ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",]


target_columns = [str("label")]
###----global remove other columns---##
continue_cols = ["I" +str(i) for i in range(1, 14)]


####continue plus id columns----###
all_feature = ID_columns[:]
all_feature.extend(continue_cols)

col_index = []
for col in  all_feature:
    col_index.append(global_columns.index(col))

#target_col =  global_columns.index(target_columns[0])


all_data = dfTrain.append(dfTest)


batch_size = 204800
epochs = 25
###----------make sure the ids max columns number---###
max_features = {}
for i in range(len(ID_columns)):
    max_features[ID_columns[i]]=(all_data[ID_columns[i]].unique().shape[0])
del all_data
gc.collect()


import numpy as np
max_features_df = pd.DataFrame(data = np.array([list(max_features.keys()),list(max_features.values())]).T,columns=['ids','max_features'],index=range(len(max_features)))
max_features = pd.merge(pd.DataFrame(ID_columns,columns=['ids']),max_features_df,on=['ids'])
max_features.max_features = max_features.max_features.astype(int)
max_features = max_features.max_features.tolist()



for i in ID_columns:
    dfTrain[i],dfTest[i] = dis_sparse(dfTrain[i].values.reshape(-1,1), \
                                      dfTest[i].values.reshape(-1,1),one_hot=False)


all_feature.append("label")
df_save = dfTrain[all_feature]
df_save['id'] = df_save.index
df_save.columns = [c.lower() for c in df_save.columns]
df_save.fillna(0.0, inplace=True)

df_save.to_csv("/data2/ruike/greenplum/cmd/criteo/train_sample_process.csv",index=False)

del dfTest
del dfTrain
gc.collect()


###_-----transofrom all the features--
'''
train_x,train_y = dfTrain[all_feature],dfTrain[target_columns]
#test_x,test_y = dfTest[all_feature],dfTest[target_columns]
del dfTest
del dfTrain
gc.collect()
#his= clf.fit(train_x.T.values,train_y.values,batch_size=batch_size,\
#                  epochs=epochs,validation_data=(test_x.T.values,test_y.values))
X = train_x.T.values
y = train_y.values


del train_x
#del validation_data
gc.collect()
'''
print("process data: {}".format(time.time() - time_begin))
time_begin = time.time()

import pdb
pdb.set_trace()

'''
import psycopg2
from io import BytesIO as StringIO
from tqdm import tqdm

conn = psycopg2.connect(database="ctr", user='ruike.xy', password='', host='127.0.0.1', port= '5432')
cursor = conn.cursor()

f = StringIO()
data=""
for i in tqdm(range(X.T.shape[0])):
    data = data + "{}\t".format(i) + "{}\t".format('{'+str(list(X.T[i]))[1:-1]+"}") + str(y[i][0]) + '\n'


#data = "0\t{9.0, 2.0, 431.0, 19.0, 4257.0, 38.0, 4.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0, 1.0, 11.0, 1.0, 1.0, 3.0, 1.0, 104.0, 1.0, 0.6, 0.9, 0.1, 2.0, 4.0, 7.0, 1.0, 8.0, 4.0, 2.0, 2.0, 2.0, 4.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}\t1\n"
f.write(data)

f.seek(0)

cursor.copy_from(f, "criteo_data",
                 columns=("id", "attributes", "class_text"),
                 sep='\t', null='\\N', size=16384)

conn.commit()   #


print("insert data to DB: {}".format(time.time() - time_begin))
time_begin = time.time()
'''

from sqlalchemy import create_engine
import pandas as pd

def save_mysql(df,table_url='criteo_data'):
    conn = create_engine('postgresql://ruike.xy@127.0.0.1:5432/ctr')
    pd.io.sql.to_sql(df,table_url,con=conn,if_exists = 'replace',index=False)
    conn.dispose()

save_mysql(df_save)

print("insert data to DB: {}".format(time.time() - time_begin))
time_begin = time.time()