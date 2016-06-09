__author__ = 'onina'

import sys
import argparse
import cPickle
from sklearn import preprocessing
import numpy as np
import os
from sklearn.decomposition import PCA

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()
def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval
def pad_frames(frames, limit):
    last_frame = frames[-1]
    padding = np.asarray([last_frame * 0.]*(limit-len(frames)))
    frames_padded = np.concatenate([frames, padding], axis=0)
    return frames_padded

def extract_frames_equally_spaced(frames, K):
    # chunk frames into 'how_many' segments and use the first frame
    # from each segment
    n_frames = len(frames)
    splits = np.array_split(range(n_frames), K)
    idx_taken = [s[0] for s in splits]
    sub_frames = frames[idx_taken]
    return sub_frames

def get_sub_frames(frames):

    K=28
    if len(frames) < K:
        frames_ = pad_frames(frames, K)
    else:
        frames_ = extract_frames_equally_spaced(frames, K)

    return frames_

def gather_feats(feats_orig,train_ids,pkl_dir):
    feats_pkl = os.path.join(pkl_dir,'train_feats.pkl')
    sampling = True
    if os.path.exists(feats_pkl):
        feats = load_pkl(feats_pkl)
    else:
        feats = feats_orig[train_ids[0]]
        if sampling:
            feats = get_sub_frames(feats)
        for key in train_ids[1:]:
            feat = feats_orig[key]
            if sampling:
                feat = get_sub_frames(feat)
            feats = np.concatenate((feats, feat), axis=0)
        print "saving concatenated feats.."
        dump_pkl(feats,feats_pkl)
    return feats

def main(argv):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'pkl_dir',
        help = 'directory where to store frames'
    )
    arg_parser.add_argument(
        'feats_pkl',
        help = 'directory where to store frames'
    )
    arg_parser.add_argument(
        'feats_out_pkl',
        help = 'video extension'
    )
    arg_parser.add_argument(
        'dataset',
        help = 'video extension'
    )
    arg_parser.add_argument(
        'type',
        help = 'video extension'
    )


    args = arg_parser.parse_args()
    pkl_dir = args.pkl_dir
    feats_pkl = args.feats_pkl
    feats_out_pkl = args.feats_out_pkl
    dataset = args.dataset
    type  = args.type


    f = open(os.path.join(pkl_dir,feats_pkl),'r')
    feats_orig = cPickle.load(f)

    # print feats_pkl

    f = open(os.path.join(pkl_dir,'train.pkl'),'r')

    if args.dataset=='youtube2text':
        train_ids = ['vid%s'%i for i in range(1,1201)]
    elif args.dataset=='vtt':
        train_ids = ['video%s'%i for i in range(0,6513)]


    if type=='std':

        feats = gather_feats(feats_orig,train_ids,pkl_dir)
        normalizer = preprocessing.Normalizer().fit(feats)
        nfeats = normalizer.transform(feats)
        std_scale = preprocessing.StandardScaler().fit(feats)

        std_feats = {}
        for key in feats_orig.keys():
            nfeats = normalizer.transform(feats_orig[key])
            std_feats[key] = std_scale.transform(nfeats)
            # std_feats[key] = nfeats

    if type=='scale':
        scaled_feats = {}
        i=0
        for key in feats_orig.keys():
            scaled_feats[key] = preprocessing.scale(feats_orig[key])
            print str(i)+'/'+str(len(feats_orig.keys()))
            i+=1

        print 'processed: '+str(len(scaled_feats))+" features "

        dump_pkl(scaled_feats,os.path.join(pkl_dir,feats_out_pkl))

    elif type=='pca':
        feats = gather_feats(feats_orig,train_ids,pkl_dir)

        pca_file = os.path.join(pkl_dir,'pca.pkl')
        pca = None
        if os.path.exists(pca_file):
            pca = load_pkl(pca_file)
        else:
            pca = PCA(n_components=1024).fit(feats)
            dump_pkl(pca,pca_file)

        pca_feats = {}
        i=0
        for key in feats_orig.keys():
            pca_feats[key] = pca.transform(feats_orig[key])
            print str(i)+'/'+str(len(feats))
            i+=1

        print 'processed: '+str(len(pca_feats))+" features "

        dump_pkl(pca_feats,os.path.join(pkl_dir,feats_out_pkl))


if __name__=='__main__':
    main(sys.argv)
