__author__ = 'oliver'


import numpy as np
import os
import pickle
import csv


def create_pickle(filename,annotations):
    vids_names = {}

    with open(filename) as csvfile:
        rows = csvfile.readlines()
        for row in rows:
            row = row.split('\t')
            vid_name = row[0]

            if len(row)>5:
                tokenized = row[5].replace('.','')
                tokenized = tokenized.replace(',','')
                tokenized = tokenized.replace('\n','')
                tokenized = tokenized.replace('\t','')
                tokenized = tokenized.lower()

                if vids_names.has_key(vid_name):
                    vids_names[vid_name] += 1
                    print 'other annots'
                else:
                    vids_names[vid_name]=1

                annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':vids_names[vid_name],'caption':row[5]}]



    return annotations,vids_names


# data_dir = '/media/sea2/datasets/challenge'
#
# annotaions = []
# vids_names = {}
#
# training_file = os.path.join(data_dir,'LSMDC15_annos_training.csv')
# train_out = open(os.path.join(data_dir,'train.pkl'), 'wb')
# annotations,vids_names = create_pickle(training_file,annotaions)
#
# valid_file = os.path.join(data_dir,'LSMDC15_annos_val.csv')
# valid_out = open(os.path.join(data_dir,'valid.pkl'), 'wb')
# annotations,vids_names = create_pickle(valid_file,annotaions)
#
# test_file = os.path.join(data_dir,'LSMDC15_annos_test.csv')
# test_out = open(os.path.join(data_dir,'test.pkl'), 'wb')
# annotations,vids_names = create_pickle(test_file,annotaions)
#
# cap_out = open(os.path.join(data_dir,'CAP.pkl'), 'wb')
# pickle.dump(annotations,cap_out)