__author__ = 'onina'

import sys
import argparse
import cPickle
from sklearn import preprocessing
import numpy as np
import os
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

    args = arg_parser.parse_args()
    pkl_dir = args.pkl_dir
    feats_pkl = args.feats_pkl
    feats_out_pkl = args.feats_out_pkl
    dataset = args.dataset

    f = open(os.path.join(pkl_dir,feats_pkl),'r')
    feats_orig = cPickle.load(f)

    # print feats_pkl

    f = open(os.path.join(pkl_dir,'train.pkl'),'r')

    if args.dataset=='youtube2text':
        train_ids = ['vid%s'%i for i in range(1,1201)]
    elif args.dataset=='vtt':
        train_ids = ['video%s'%i for i in range(0,6513)]


    feats = feats_orig[train_ids[0]]
    # for key in train_ids[1:]:
    #     feat = feats_orig[key]
    #     feats = np.concatenate((feats, feat), axis=0)
    #
    # print 'original:'+str(len(feats))

    normalizer = preprocessing.Normalizer().fit(feats)
    # nfeats = normalizer.transform(feats)
    # std_scale = preprocessing.StandardScaler().fit(feats)

    std_feats = {}
    for key in feats_orig.keys():
        nfeats = normalizer.transform(feats_orig[key])
        # std_feats[key] = std_scale.transform(nfeats)
        std_feats[key] = nfeats

    print 'processed: '+str(len(std_feats))

    f = open(os.path.join(pkl_dir,feats_out_pkl), 'wb')
    try:
        cPickle.dump(std_feats, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


if __name__=='__main__':
    main(sys.argv)
