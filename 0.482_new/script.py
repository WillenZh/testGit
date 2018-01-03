# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Ensemble of two public kernels
# For the median-based kernel, I add a multiplier of 0.95 to the original result.
# Median-based from Paulo Pinto: https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529
# LGBM from Ceshine Lee: https://www.kaggle.com/ceshine/lgbm-starter

filelist = ['./Surprise2.csv', './sub_hrm_mean.csv']

outs = [pd.read_csv(f, index_col=0) for f in filelist]
concat_df = pd.concat(outs, axis=1)
concat_df.columns = ['submission1', 'submission2']
#concat_df["visitors"] = concat_df.mean(axis=1)
concat_df["visitors"] = ((concat_df['submission1']*.85) + (concat_df['submission2']*.15)*1.15) 
#concat_df["visitors"] = concat_df['submission1']
concat_df[["visitors"]].to_csv("ensemble.csv")