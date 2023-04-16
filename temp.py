import glob
import shutil
import pandas as pd
import random
import time

#ratio = 0.8
#path = r'C:\Users\SHIN-DESKTOP\Documents\remote\my_korean_hate_speech\data\korean-hate-speech\labeled\train.tsv'
#df = pd.read_csv(path, sep='\t')
#df = df.sample(frac=1).reset_index(drop=True)
#df_train = df[:int(len(df) * ratio)]
#df_test = df[int(len(df) * ratio):]
#df_train.to_csv('new_train.tsv', sep='\t', index=False)
#df_test.to_csv('new_test.tsv', sep='\t', index=False)


d = {'0': 'none', '1': 'off', '2': 'hate'}
path = r'C:\Users\SHIN-DESKTOP\Documents\remote\my_korean_hate_speech\log'
"""
files = glob.glob(path+'//'+'*.png')
for file in files:
    name = file.split('\\')[-1]
    name_wo_ext = name.split('.')[0]
    x = name_wo_ext.split('_')
    y = []
    for i in x:
        if i in d.keys():
            y.append(d[i])
        else:
            y.append(i)

    subpath = y[1]+'_'+y[2]
    dest_file = path + '\\' + subpath + '\\' + name
    shutil.move(file, dest_file)
"""
