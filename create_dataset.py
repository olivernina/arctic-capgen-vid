__author__ = 'oliver'

import os,sys
from create_lists import create_pickle
import process_frames
# import cPickle as pickle
import pickle
import process_features
import argparse


def main(params):

    # sys.stdout = open('out.log','w')

    data_root = params['data_dir']
    data_dir =os.path.join(params['data_dir'],'challenge')
    dst_dir = os.path.join(params['data_dir'],'frames_chal')
    pkl_dir = params['pkl_dir']

    test_mode = params['test']

    annotations = {}

    all_vids = []

    if test_mode:
        training_file = os.path.join(data_dir,'LSMDC15_annos_training_small.csv')
    else:
        training_file = os.path.join(data_dir,'LSMDC15_annos_training.csv')

    annotations,vids_names = create_pickle(training_file,annotations,)
    training_list = vids_names.keys()
    train_out = open(os.path.join(pkl_dir,'train.pkl'), 'wb')
    pickle.dump(training_list,train_out)
    all_vids = all_vids + training_list

    # sys.stdout.flush()

    if test_mode:
        valid_file = os.path.join(data_dir,'LSMDC15_annos_val_small.csv')
    else:
        valid_file = os.path.join(data_dir,'LSMDC15_annos_val.csv')

    valid_out = open(os.path.join(pkl_dir,'valid.pkl'), 'wb')
    annotations,vids_names = create_pickle(valid_file,annotations)
    valid_list = vids_names.keys()
    pickle.dump(valid_list,valid_out)
    all_vids = all_vids + valid_list

    if test_mode:
        test_file = os.path.join(data_dir,'LSMDC15_annos_test_small.csv')
    else:
        test_file = os.path.join(data_dir,'LSMDC15_annos_test.csv')

    test_out = open(os.path.join(pkl_dir,'test.pkl'), 'wb')
    annotations,vids_names = create_pickle(test_file,annotations)
    test_list = vids_names.keys()
    pickle.dump(test_list,test_out)
    all_vids = all_vids + test_list

    if test_mode:
        blindtest_file = os.path.join(data_dir,'LSMDC15_annos_blindtest_small.csv')
    else:
        blindtest_file = os.path.join(data_dir,'LSMDC15_annos_blindtest.csv')

    blindtest_out = open(os.path.join(pkl_dir,'blindtest.pkl'), 'wb')
    annotations,vids_names = create_pickle(blindtest_file,annotations)
    blindtest_list = vids_names.keys()
    pickle.dump(blindtest_list,blindtest_out)
    all_vids = all_vids + test_list

    cap_out = open(os.path.join(pkl_dir,'CAP.pkl'), 'wb')
    pickle.dump(annotations,cap_out)

    worddict = {}
    word_idx = 2
    for a in annotations:
        ann = annotations[a][0]
        tokens = ann['tokenized'].split()
        for token in tokens:
            if token not in ['','\t','\n',' ']:
                if not worddict.has_key(token):
                    worddict[token]=word_idx
                    word_idx+=1

    worddict_out = open(os.path.join(pkl_dir,'worddict.pkl'), 'wb')
    pickle.dump(worddict,worddict_out)

    # sys.stdout.flush()

    vid_frames = []

    for file in all_vids:
        k = file.rfind("_")
        movie_dir = file[:k]
        video_name = file+'.avi'
        src_dir = os.path.join(data_dir,movie_dir)
        frames_dir = process_frames.get_frames(src_dir,dst_dir,video_name)
        vid_frames.append(frames_dir)


    process_features.run(vid_frames,data_root,pkl_dir)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/sea2/datasets')
    parser.add_argument('-t','--test',dest = 'test',type=int,default=0, help='perform small test')
    parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='/media/sea2/datasets/challenge')
    args = parser.parse_args()
    params = vars(args)

    main(params)