__author__ = 'oliver'

import os,sys
import process_frames
import pickle
import argparse
import nltk
import common
import socket
import json
import numpy as np

host = socket.gethostname()

if host == 'moroni':
    import process_features



def get_annots_lsmdc(filename,annotations):
    vids_names = {}

    with open(filename) as csvfile:
        rows = csvfile.readlines()
        for row in rows:
            row = row.split('\t')
            vid_name = row[0]

            if len(row)>5:
                ocaption = row[5]

                ocaption = ocaption.replace('\n','')
                udata=ocaption.decode("utf-8")
                caption = udata.encode("ascii","ignore")

                tokens = nltk.word_tokenize(caption)
                tokenized = ' '.join(tokens)
                tokenized = tokenized.lower()

                if vids_names.has_key(vid_name):
                    vids_names[vid_name] += 1
                    print 'other annots'
                else:
                    if not os.path.exists('/media/onina/sea2/datasets/lsmdc/features_chal/'+vid_name):
                        print 'features not found'
                    vids_names[vid_name]=1

                annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':vids_names[vid_name],'caption':ocaption}]



    return annotations,vids_names


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
        caption = caption.replace('\n','')

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

        annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':vids_names[vid_name],'caption':caption}]



    return annotations,vids_names

def get_annots_vtt(filename,annotations):
    vids_train = []
    vids_val = []
    all_vids = {}
    with open(filename) as data_file:
        data = json.load(data_file)


    for sent in data['sentences']:#[0:1000]:

        vid_name = sent['video_id']

        if not all_vids.has_key(vid_name):
            all_vids[vid_name]=1
        else:
            all_vids[vid_name]+=1

        ocaption = sent['caption']
        print vid_name
        print ocaption

        # ocaption = ocaption.replace('\n','')
        ocaption = ocaption.strip()
        # udata=ocaption.decode("utf-8")
        # caption = udata.encode("ascii","ignore")

        tokens = nltk.word_tokenize(ocaption)
        tokenized = ' '.join(tokens)
        tokenized = tokenized.lower()

        vid_id = int(vid_name.split('video')[1])


        cap_id = -1
        if annotations.has_key(vid_name):
            cap_id = str(len(annotations[vid_name]))
            annotations[vid_name].append({'tokenized':tokenized,'image_id':vid_name,'cap_id':cap_id,'caption':ocaption})
        else:
            annotations[vid_name]= []
            cap_id = str(0)
            annotations[vid_name].append({'tokenized':tokenized,'image_id':vid_name,'cap_id':cap_id,'caption':ocaption})


        if data['videos'][vid_id]['split'] == 'train':

            # if vids_train.has_key(vid_name):
            #     vids_train[vid_name] += 1
            # else:
            #     vids_train[vid_name]=1
            vid_train =vid_name + '_' + cap_id
            vids_train.append(vid_train)

        elif data['videos'][vid_id]['split'] == 'validate':
            # if vids_val.has_key(vid_name):
            #     vids_val[vid_name] += 1
            # else:
            #     vids_val[vid_name]=1
            vid_val = vid_name + '_' + cap_id
            vids_val.append(vid_val)
        else:
            print "not a split"

    return annotations,vids_train,vids_val,all_vids.keys()




def lsmdc_old(params):

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
        caps = annotations[a]

        for cap in caps:
            tokens = cap['tokenized'].split()
            for token in tokens:
                if token not in ['','\t','\n',' ']:
                    if not worddict.has_key(token):
                        worddict[token]=word_idx
                        word_idx+=1

    return worddict

def get_frames_vtt(all_vids,vid_dir,dst_dir):
    vid_frames = []

    for file in all_vids:
        # k = file.rfind("_")
        # movie_dir = file[:k]
        video_name = file+'.mp4'


        frames_dir = process_frames.get_frames(vid_dir,'',dst_dir,video_name)
        vid_frames.append(frames_dir)

    return vid_frames

def get_frames(all_vids,vid_dir,dst_dir):
    vid_frames = []

    for file in all_vids:
        k = file.rfind("_")
        movie_dir = file[:k]
        video_name = file+'.avi'


        frames_dir = process_frames.get_frames(vid_dir,movie_dir,dst_dir,video_name)
        vid_frames.append(frames_dir)

    return vid_frames


def get_frames_mpii(all_vids,vid_dir,dst_dir):
    vid_frames = []

    for i,file in enumerate(all_vids):
        k = file.rfind("_")
        movie_dir = file[:k]
        video_name = file+'.avi'


        frames_dir = process_frames.get_frames_mpii(vid_dir,movie_dir,dst_dir,video_name)
        vid_frames.append(frames_dir)

    return vid_frames


def get_frames_ysvd(all_vids,vid_dir,dst_dir):
    vid_frames = []

    for file in all_vids:

        movie_dir = file
        video_name = file+'.mp4'


        frames_dir = process_frames.get_frames(vid_dir,'',dst_dir,video_name)
        vid_frames.append(frames_dir)

    return vid_frames

def lsmdc(params):

    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']

    test_mode = params['test']
    if test_mode:
        train_list_path = 'lists/small/LSMDC15_annos_training_small.csv'
        valid_list_path = 'lists/small/LSMDC15_annos_val_small.csv'
        test_list_path = 'lists/small/LSMDC15_annos_test_small.csv'
    else:
        train_list_path = 'lists/LSMDC15_annos_training.csv'
        valid_list_path = 'lists/LSMDC15_annos_val.csv'
        test_list_path = 'lists/LSMDC15_annos_test.csv'

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []


    train_path = os.path.join(pkl_dir,'train.pkl')
    if not os.path.exists(train_path):
        train_file = os.path.join(data_dir,train_list_path)
        annotations,vids_names = get_annots_lsmdc(train_file,annotations)
        training_list = vids_names.keys()
        common.dump_pkl(training_list,train_path)
    else:
        training_list = common.load_pkl(train_path)


    all_vids = all_vids + training_list


    valid_path = os.path.join(pkl_dir,'valid.pkl')
    if not os.path.exists(valid_path):
        valid_file = os.path.join(data_dir,valid_list_path)
        annotations,vids_names = get_annots_lsmdc(valid_file,annotations)
        valid_list = vids_names.keys()
        common.dump_pkl(valid_list,valid_path)
    else:
        valid_list = common.load_pkl(valid_path)

    all_vids = all_vids + valid_list


    test_path = os.path.join(pkl_dir,'test.pkl')
    if not os.path.exists(test_path):
        test_file = os.path.join(data_dir,test_list_path)
        annotations,vids_names = get_annots_lsmdc(test_file,annotations)
        test_list = vids_names.keys()
        common.dump_pkl(test_list,test_path)
    else:
        test_list = common.load_pkl(test_path)

    all_vids = all_vids + test_list

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
       common.dump_pkl(annotations,cap_path)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations,dict_path)
        common.dump_pkl(worddict,dict_path)


    if host != 'moroni' or test_mode:
        feats_paths = list()
        feats = {}
        for video in all_vids:
            feat_path =  os.path.join(feats_dir,video)
            feats_paths.append(feat_path)

            if os.path.exists(feat_path):
                feat = np.load(feat_path)
                feats[video]=feat
                print('features already extracted '+feat_path)
            else:
                print "feature not found "+feat_path
                sys.exit(0)

        # features = process_features.run(vid_paths,feats_dir,frames_dir)
        feats_pkl_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
        common.dump_pkl(feats,feats_pkl_path)
    else:

        # all_vids = all_vids[46000:-1]
        vid_frames = get_frames(all_vids,video_dir,frames_dir)
        features = process_features.run(vid_frames,feats_dir,frames_dir,'.avi') # We don't save the FEAT file because it requires to much memory TODO

    print('done creating dataset')

def mpii(params):

    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']

    if params['test']:

        f = open(os.path.join(data_dir,'test','downloadLinksAvi.txt'),'rb')
        files = f.readlines()
        f.close()
        f = open(os.path.join(data_dir,'test','annotations-someone.csv'),'rb')
        annots = f.readlines()
        f.close()
        f = open(os.path.join(data_dir,'test','dataSplit.txt'),'rb')
        splits_file = f.readlines()
        splits = {}

    elif host == 'moroni':

        f = open(os.path.join(data_dir,'downloadLinksAvi.txt'),'rb')
        files = f.readlines()
        f.close()
        f = open(os.path.join(data_dir,'annotations-someone.csv'),'rb')
        annots = f.readlines()
        f.close()
        f = open(os.path.join(data_dir,'dataSplit.txt'),'rb')
        splits_file = f.readlines()
        splits = {}




    annotations = {}
    train_clip_names = []
    valid_clip_names = []
    test_clip_names = []

    train_path = os.path.join(pkl_dir,'train.pkl')
    if not os.path.exists(train_path):
        for line in splits_file:
            film_name = line.split('\t')[0]
            split = line.split('\t')[1]
            splits[film_name] =split.replace('\r\n','')

        for i,file in enumerate(files):
            parts = file.split('/')

            film_name = parts[6]
            clip_name = parts[7].replace('\n','')
            clip_name = clip_name.split('.avi')[0]
            caption = annots[i].split('\t')[1]
            caption = caption.replace('\n','')


            udata=caption.decode("utf-8")
            caption = udata.encode("ascii","ignore")

            tokens = nltk.word_tokenize(caption)
            tokenized = ' '.join(tokens)
            tokenized = tokenized.lower()

            annotations[clip_name]=[{'tokenized':tokenized,'image_id':clip_name,'cap_id':1,'caption':caption}]

            if splits[film_name]=='training':
                train_clip_names.append(clip_name)
            elif splits[film_name]=='validation':
                valid_clip_names.append(clip_name)
            elif splits[film_name]=='test':
                test_clip_names.append(clip_name)



    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []


    train_path = os.path.join(pkl_dir,'train.pkl')
    if not os.path.exists(train_path):
        common.dump_pkl(train_clip_names,train_path)
    else:
        train_clip_names = common.load_pkl(train_path)

    all_vids = all_vids + train_clip_names


    valid_path = os.path.join(pkl_dir,'valid.pkl')
    if not os.path.exists(valid_path):
        common.dump_pkl(valid_clip_names,valid_path)
    else:
        valid_clip_names = common.load_pkl(valid_path)

    all_vids = all_vids + valid_clip_names


    test_path = os.path.join(pkl_dir,'test.pkl')
    if not os.path.exists(test_path):
        common.dump_pkl(test_clip_names,test_path)
    else:
        test_clip_names = common.load_pkl(test_path)

    all_vids = all_vids + test_clip_names

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
       common.dump_pkl(annotations,cap_path)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations,dict_path)
        common.dump_pkl(worddict,dict_path)


    if host != 'moroni' or params['test']:
        feats_paths = list()
        feats = {}
        for video in all_vids:
            feat_path =  os.path.join(feats_dir,video)
            feats_paths.append(feat_path)

            if os.path.exists(feat_path):
                feat = np.load(feat_path)
                feats[video]=feat
                print('features already extracted '+feat_path)
            else:
                print "feature not found "+feat_path
                sys.exit(0)

        # features = process_features.run(vid_paths,feats_dir,frames_dir)
        feats_pkl_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
        common.dump_pkl(feats,feats_pkl_path)
    else:

        # all_vids = all_vids[68370:-1]
        vid_frames = get_frames_mpii(all_vids,video_dir,frames_dir)
        frames_dir = '/media/onina/sea2/datasets/mpii/frames'
        features = process_features.run_mpii(vid_frames,feats_dir,frames_dir, '.avi') # We don't save the FEAT file because it requires to much memory TODO

    print('done creating dataset')


def mvad(params):


    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']

    test_mode = params['test']
    if test_mode:
        train_list_path = 'small/TrainList.txt'
        train_corpus_path = 'small/TrainCorpus.txt'
        valid_list_path = 'small/ValidList.txt'
        valid_corpus_path = 'small/ValidCorpus.txt'
        test_list_path = 'small/TestList.txt'
        test_corpus_path = 'small/TestCorpus.txt'
    else:
        train_list_path = 'TrainList.txt'
        train_corpus_path = 'TrainCorpus.txt'
        valid_list_path = 'ValidList.txt'
        valid_corpus_path = 'ValidCorpus.txt'
        test_list_path = 'TestList.txt'
        test_corpus_path = 'TestCorpus.txt'

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []


    train_path = os.path.join(pkl_dir,'train.pkl')
    if not os.path.exists(train_path):
        train_file = os.path.join(data_dir,train_list_path)
        train_corpus = os.path.join(data_dir,train_corpus_path)
        annotations,vids_names = get_annots_mvad(train_file,train_corpus,annotations)
        training_list = vids_names.keys()
        common.dump_pkl(training_list,train_path)
    else:
        training_list = common.load_pkl(train_path)

    all_vids = all_vids + training_list


    valid_path = os.path.join(pkl_dir,'valid.pkl')
    if not os.path.exists(valid_path):
        valid_file = os.path.join(data_dir,valid_list_path)
        valid_corpus = os.path.join(data_dir,valid_corpus_path)
        annotations,vids_names = get_annots_mvad(valid_file,valid_corpus,annotations)
        valid_list = vids_names.keys()
        common.dump_pkl(valid_list,valid_path)
    else:
        valid_list = common.load_pkl(valid_path)

    all_vids = all_vids + valid_list


    test_path = os.path.join(pkl_dir,'test.pkl')
    if not os.path.exists(test_path):
        test_file = os.path.join(data_dir,test_list_path)
        test_corpus = os.path.join(data_dir,test_corpus_path)
        annotations,vids_names = get_annots_mvad(test_file,test_corpus,annotations)
        test_list = vids_names.keys()
        common.dump_pkl(test_list,test_path)
    else:
        test_list = common.load_pkl(test_path)

    all_vids = all_vids + test_list

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
       common.dump_pkl(annotations,cap_path)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations,dict_path)
        common.dump_pkl(worddict,dict_path)





    if host != 'moroni' or test_mode:
        feats_paths = list()
        feats = {}
        for video in all_vids:
            feat_path =  os.path.join(feats_dir,video)
            feats_paths.append(feat_path)

            if os.path.exists(feat_path):
                feat = np.load(feat_path)
                feats[video]=feat
                print('features already extracted '+feat_path)
            else:
                print "feature not found "+feat_path
                sys.exit(0)

        # features = process_features.run(vid_paths,feats_dir,frames_dir)
        feats_pkl_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
        common.dump_pkl(feats,feats_pkl_path)
    else:

        # all_vids = all_vids[46000:-1]
        vid_frames = get_frames_mvad(all_vids,video_dir,frames_dir)
        features = process_features.run(vid_frames,feats_dir,frames_dir) # We don't save the FEAT file because it requires to much memory TODO

    print('done creating dataset')

def get_human_annotations(data_dir):
    hannot_path = os.path.join(data_dir,'human_annotations','HumanCaps.csv')
    import csv

    hannot = {}
    with open(hannot_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in spamreader:
            print ', '.join(row)
            hannot[row[0]] = row[1]
    return hannot

def tokenize_cap(caption):
    udata=caption.decode("utf-8")
    caption = udata.encode("ascii","ignore")

    tokens = nltk.word_tokenize(caption)
    tokenized = ' '.join(tokens)
    tokenized = tokenized.lower()
    return tokenized

def ysvd(params):

    data_root = params['data_dir']
    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']


    desc_path = os.path.join(data_dir,'description')

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)


    annotations = {}

    vids_names = {}

    hannots = get_human_annotations(data_dir)


    for file in os.listdir(video_dir):

        vid_name = file.split('.')[0]

        print vid_name

        desc_file = os.path.join(desc_path,vid_name+'.description')
        f = open(desc_file,'r')
        desc = f.read()


        caption = desc.split('.')[0]

        tokenized = tokenize_cap(caption)
        # caption = desc[0:100]

        caption2 = hannots[vid_name]
        tokenized2 = tokenize_cap(caption2)


        if vids_names.has_key(vid_name):
            vids_names[vid_name] += 1
            print 'other annots, there should be only 1. TODO'
            # sys.exit(0)
        else:
            # if not os.path.exists('/media/onina/sea2/datasets/lsmdc/features_chal/'+vid_name):
            #     print 'features not found'
            vids_names[vid_name]=1

        annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':0,'caption':caption},{'tokenized':tokenized2,'image_id':vid_name,'cap_id':1,'caption':caption2} ]


    all_vids = vids_names.keys()

    vids_list_path = os.path.join(pkl_dir,'all_vids.pkl')
    if not os.path.exists(vids_list_path):
       common.dump_pkl(all_vids,vids_list_path)

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
       common.dump_pkl(annotations,cap_path)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations,dict_path)
        common.dump_pkl(worddict,dict_path)

    vid_frames = get_frames_ysvd(all_vids,video_dir,frames_dir)

    features = process_features.run(vid_frames,feats_dir,frames_dir)
    feats_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
    common.dump_pkl(features,feats_path)



def vtt(params):

    data_dir =params['data_dir']
    video_dir = params['video_dir']
    frames_dir = params['frames_dir']
    pkl_dir = params['pkl_dir']
    feats_dir = params['feats_dir']

    test_mode = params['test']
    if test_mode:
        train_val_list_path = 'train_val_annotation/train_val_videodatainfo.json'

    else:
        train_val_list_path = 'train_val_annotation/train_val_videodatainfo.json'


    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = None


    train_path = os.path.join(pkl_dir,'train.pkl')
    valid_path = os.path.join(pkl_dir,'valid.pkl')
    test_path = os.path.join(pkl_dir,'test.pkl')

    if not os.path.exists(train_path):
        train_val_file = os.path.join(data_dir,train_val_list_path)
        annotations,vids_train,vids_val,all_vids = get_annots_vtt(train_val_file,annotations)

        # train_list = vids_train.keys()
        common.dump_pkl(vids_train,train_path)
        # all_vids = all_vids + vids_train

        # valid_list = vids_val.keys()
        common.dump_pkl(vids_val,valid_path)
        # all_vids = all_vids + vids_val

        # test_list = vids_val.keys()
        common.dump_pkl(vids_val,test_path)
        # all_vids = all_vids + test_list

    else:
        train_list = common.load_pkl(train_path)
        valid_list = common.load_pkl(valid_path)
        test_list = common.load_pkl(test_path)

    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    if not os.path.exists(cap_path):
       common.dump_pkl(annotations,cap_path)
    else:
        annotations = common.load_pkl(cap_path)

    dict_path = os.path.join(pkl_dir,'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations,dict_path)
        common.dump_pkl(worddict,dict_path)
    else:
        worddict = common.load_pkl(dict_path)

    # if host != 'moroni' or test_mode:
    #     feats_paths = list()
    #     feats = {}
    #     for video in all_vids:
    #         feat_path =  os.path.join(feats_dir,video)
    #         feats_paths.append(feat_path)
    #
    #         if os.path.exists(feat_path):
    #             feat = np.load(feat_path)
    #             feats[video]=feat
    #             print('features already extracted '+feat_path)
    #         else:
    #             print "feature not found "+feat_path
    #             sys.exit(0)
    #
    #     # features = process_features.run(vid_paths,feats_dir,frames_dir)
    #     feats_pkl_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
    #     common.dump_pkl(feats,feats_pkl_path)
    # else:

        # all_vids = all_vids[46000:-1]
    feats_path = os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl')
    if not os.path.exists(feats_path):
        vid_frames = get_frames_vtt(all_vids,video_dir,frames_dir)
        features = process_features.run(vid_frames,feats_dir,frames_dir,'.mp4') # We don't save the FEAT file because it requires to much memory TODO
        common.dump_pkl(features,feats_path)





    print('done creating dataset')

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea2/datasets/mpii')
    # parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/videos')
    # parser.add_argument('-frame','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/frames_chal')
    # parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea2/datasets/mpii/features')
    # parser.add_argument('-t','--test',dest = 'test',type=int,default=0, help='perform small test')
    # parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/mpii/')
    # parser.add_argument('-dbname','--dbname',dest ='dbname',type=str,default='mpii')


    # parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea2/datasets/mvad')
    # parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/videos')
    # parser.add_argument('-frame','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/frames_chal')
    # parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/features_chal')
    # parser.add_argument('-t','--test',dest = 'test',type=int,default=0, help='perform small test')
    # parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/mvad/')
    # parser.add_argument('-dbname','--dbname',dest ='dbname',type=str,default='mvad')

    # parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea1/datasets/ysvd')
    # parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea1/datasets/ysvd/videos')
    # parser.add_argument('-f','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea1/datasets/ysvd/frames')
    # parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea1/datasets/ysvd/features')
    # parser.add_argument('-t','--test',dest = 'test',type=int,default=1, help='perform small test')
    # parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/ysvd/')
    # parser.add_argument('-dbname','--dbname',dest ='dbname',type=str,default='ysvd')


    # parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea2/datasets/lsmdc')
    # parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/videos')
    # parser.add_argument('-frame','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/frames_chal')
    # parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea2/datasets/lsmdc/features_chal')
    # parser.add_argument('-t','--test',dest = 'test',type=int,default=0, help='perform small test')
    # parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/lsmdc/')
    # parser.add_argument('-dbname','--dbname',dest ='dbname',type=str,default='lsmdc')


    parser.add_argument('-d','--data_dir',dest ='data_dir',type=str,default='/media/onina/sea2/datasets/vtt')
    parser.add_argument('-v','--video_dir',dest ='video_dir',type=str,default='/media/onina/sea2/datasets/vtt/TrainValVideo')
    parser.add_argument('-frame','--frames_dir',dest ='frames_dir',type=str,default='/media/onina/sea2/datasets/vtt/frames')
    parser.add_argument('-feat','--feats_dir',dest ='feats_dir',type=str,default='/media/onina/sea2/datasets/vtt/features')
    parser.add_argument('-t','--test',dest = 'test',type=int,default=1, help='perform small test')
    parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='./data/vtt/')
    parser.add_argument('-dbname','--dbname',dest ='dbname',type=str,default='vtt')


    args = parser.parse_args()
    params = vars(args)

    if params['dbname'] == 'mvad':
        mvad(params)
    if params['dbname'] == 'ysvd':
        ysvd(params)
    if params['dbname'] == 'mpii':
        mpii(params)
    if params['dbname'] == 'lsmdc':
        lsmdc(params)
    if params['dbname'] == 'vtt':
        vtt(params)