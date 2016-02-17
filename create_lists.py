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
                    if not os.path.exists('/media/onina/sea2/datasets/lsmdc/features_chal/'+vid_name):
                        print 'features not found'
                    vids_names[vid_name]=1

                annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':vids_names[vid_name],'caption':row[5]}]



    return annotations,vids_names
