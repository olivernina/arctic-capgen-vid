__author__ = 'oliver'



import sys
import os.path
import argparse

import numpy as np
from scipy.misc import imread, imresize
import scipy.io

import cPickle as pickle


caffepath = 'caffe/python'
sys.path.append(caffepath)

import caffe

def predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    #features = out[net.outputs[0]].squeeze(axis=(2,3))
    features = out[net.outputs[0]]

    return features


def batch_predict(filenames, net):
    """
    Get the features for all images from filenames using a network

    Inputs:
    filenames: a list of names of image files

    Returns:
    an array of feature vectors for the images in that file
    """

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:,0,0] #for googlenet
            # allftrs[i+j,:] = ftrs[j,:] #for vgg network

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs


def get_features(src_dir,dst_dir,video_dir,net):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_path =  os.path.join(src_dir,video_dir)
    if os.path.exists(src_path):
        dst_path = os.path.join(dst_dir, video_dir.split('.avi')[0])
        if not os.path.exists(dst_path):

            # base_dir = os.path.dirname(files)
            frames = os.listdir(src_path)
            frames.sort()

            filenames = [os.path.join(src_path,x) for x in frames]

            print 'processing '+ dst_path+' '+str(len(filenames))+' frames'
            allftrs = batch_predict(filenames, net)
            feat_file = open(dst_path, 'wb')
            np.save(feat_file,allftrs)
            print ' features created'
            return allftrs

        else:
            print('features already extracted')
        return dst_path
    else:
        print('video: '+src_path+' doesn\'t exist')
        return False


def run(vid_frames,data_dir, pkl_dir):

    caffe.set_mode_gpu()


    model_def = 'caffe/models/bvlc_googlenet/deploy_video.prototxt'
    model = 'caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

    # --out vgg_feats.mat'+' --spv '+str(samples_per_video)
    net = caffe.Net(model_def, model, caffe.TEST)
    #caffe.set_phase_test()


    feats = {}

    for i,files in enumerate(vid_frames):
        feats_dir =os.path.join(data_dir,'features_chal')
        feat_filename = files.split('/')[-1].split('.avi')[0]
        feat_file_path = os.path.join(feats_dir,feat_filename)

        if os.path.exists(feat_file_path):
            feat = np.load(feat_file_path)
            feats[feat_filename]=feat
            print('features already extracted '+feat_file_path)
        else:
            frames_dir = os.path.join(data_dir,'frames_chal')
            feat = get_features(frames_dir,feats_dir,files.split('/')[-1],net)
            feats[feat_filename]=feat

        # sys.stdout.flush()
        print str(i)+'/'+str(len(vid_frames))



    feats_out = open(os.path.join(pkl_dir,'FEAT_key_vidID_value_features.pkl'), 'wb')
    pickle.dump(feats,feats_out)


    print('done')

