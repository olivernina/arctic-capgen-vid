__author__ = 'oliver'

import os,sys
from create_lists import create_pickle
import process_frames
# import cPickle as pickle
import pickle
import process_features
import argparse
import nltk

def get_annots_mvad(list,corpus,annotations):
    vids_names = {}

    list_file = open(list,'rw')
    corpus_file = open(corpus,'rw')
    rows = list_file.readlines()
    annots = corpus_file.readlines()

    for i,row in enumerate(rows):


            # row = row.split('\t')
        vid_name = row.split('/')[-1].split('.')[0]
        caption = annots[i]

        udata=caption.decode("utf-8")
        caption = udata.encode("ascii","ignore")

        tokens = nltk.word_tokenize(caption)
        tokenized = ' '.join(tokens)
        tokenized = tokenized.lower()

        if vids_names.has_key(vid_name):
            vids_names[vid_name] += 1
            print 'other annots, there should be only 1'
            sys.exit(0)
        else:
            if not os.path.exists('/media/onina/sea2/datasets/lsmdc/features_chal/'+vid_name):
                print 'features not found'
            vids_names[vid_name]=1

        annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':vids_names[vid_name],'caption':row[5]}]



    return annotations,vids_names


def lsmdc(params):

    # sys.stdout = open('out.log','w')

    data_root = params['data_dir']
    data_dir =params['data_dir']
    dst_dir = os.path.join(params['data_dir'],'frames_chal')
    pkl_dir = params['pkl_dir']

    test_mode = params['test']

    annotations = {}

    all_vids = []

    if test_mode:
        training_file = os.path.join(data_dir,'list/smaller/LSMDC15_annos_training_small.csv')
    else:
        training_file = os.path.join(data_dir,'LSMDC15_annos_training.csv')

    annotations,vids_names = create_pickle(training_file,annotations,)
    training_list = vids_names.keys()
    train_out = open(os.path.join(pkl_dir,'train.pkl'), 'wb')
    pickle.dump(training_list,train_out)
    all_vids = all_vids + training_list

    # sys.stdout.flush()

    if test_mode:
        valid_file = os.path.join(data_dir,'list/smaller/LSMDC15_annos_val_small.csv')
    else:
        valid_file = os.path.join(data_dir,'LSMDC15_annos_val.csv')

    valid_out = open(os.path.join(pkl_dir,'valid.pkl'), 'wb')
    annotations,vids_names = create_pickle(valid_file,annotations)
    valid_list = vids_names.keys()
    pickle.dump(valid_list,valid_out)
    all_vids = all_vids + valid_list

    if test_mode:
        test_file = os.path.join(data_dir,'list/smaller/LSMDC15_annos_test_small.csv')
    else:
        test_file = os.path.join(data_dir,'LSMDC15_annos_test.csv')

    test_out = open(os.path.join(pkl_dir,'test.pkl'), 'wb')
    annotations,vids_names = create_pickle(test_file,annotations)
    test_list = vids_names.keys()
    pickle.dump(test_list,test_out)
    all_vids = all_vids + test_list

    if test_mode:
        blindtest_file = os.path.join(data_dir,'list/smaller/LSMDC15_annos_blindtest_small.csv')
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
        src_dir = os.path.join(data_dir,'videos',movie_dir)
        frames_dir = process_frames.get_frames(src_dir,dst_dir,video_name)
        vid_frames.append(frames_dir)


    process_features.run(vid_frames,data_root,pkl_dir)

def create_dictionary(annotations,pkl_dir):
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


def get_frames(all_vids,vid_dir,dst_dir):
    vid_frames = []

    for file in all_vids:
        k = file.rfind("_")
        movie_dir = file[:k]
        video_name = file+'.avi'

        src_dir = os.path.join(vid_dir,movie_dir)
        frames_dir = process_frames.get_frames(src_dir,dst_dir,video_name)
        vid_frames.append(frames_dir)

    return vid_frames

def mvad(params):

    data_root = params['data_dir']
    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']

    test_mode = params['test']
    if test_mode:
        train_list_path = 'small/TrainList.txt'
        train_corpus_path = 'small/TrainCorpus.txt'
        valid_path = 'small/ValidList.txt'
        valid_corpus_path = 'small/ValidCorpus.txt'
        test_path = 'small/TestList.txt'
        test_corpus_path = 'small/TestCorpus.txt'
    else:
        train_list_path = 'TrainList.txt'
        train_corpus_path = 'TrainCorpus.txt'
        valid_path = 'ValidList.txt'
        valid_corpus_path = 'ValidCorpus.txt'
        test_path = 'TestList.txt'
        test_corpus_path = 'TestCorpus.txt'

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []

    train_file = os.path.join(data_dir,train_list_path)
    train_corpus = os.path.join(data_dir,train_corpus_path)
    annotations,vids_names = get_annots_mvad(train_file,train_corpus,annotations)
    training_list = vids_names.keys()
    train_path = os.path.join(pkl_dir,'train.pkl')
    if not os.path.exists(train_path):
        train_out = open(train_path, 'wb')
        pickle.dump(training_list,train_out)

    all_vids = all_vids + training_list


    valid_file = os.path.join(data_dir,valid_path)
    valid_corpus = os.path.join(data_dir,valid_corpus_path)
    annotations,vids_names = get_annots_mvad(valid_file,valid_corpus,annotations)
    valid_list = vids_names.keys()
    valid_path = os.path.join(pkl_dir,'valid.pkl')
    if not os.path.exists(valid_path):
        valid_out = open(valid_path, 'wb')
        pickle.dump(valid_list,valid_out)

    all_vids = all_vids + valid_list

    test_file = os.path.join(data_dir,test_path)
    test_corpus = os.path.join(data_dir,test_corpus_path)
    annotations,vids_names = get_annots_mvad(test_file,test_corpus,annotations)
    test_list = vids_names.keys()
    test_path = os.path.join(pkl_dir,'test.pkl')
    if not os.path.exists(test_path):
        test_out = open(test_path, 'wb')
        pickle.dump(test_list,test_out)


    all_vids = all_vids + test_list

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
        cap_out = open(cap_path, 'wb')
        pickle.dump(annotations,cap_out)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        create_dictionary(annotations,pkl_dir)

    all_vids = all_vids[30000:-1]
    vid_frames = get_frames(all_vids,video_dir,frames_dir)

    features = process_features.run(vid_frames,feats_dir,frames_dir)


    # feats_out = open(os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl'), 'wb')
    # pickle.dump(features,feats_out)


    print('done creating dataset')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea2/datasets/mvad')
    parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/videos')
    parser.add_argument('-f','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/frames_chal')
    parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/features_chal')
    parser.add_argument('-t','--test',dest = 'test',type=int,default=0, help='perform small test')
    parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/mvad/')
    args = parser.parse_args()
    params = vars(args)

    mvad(params)