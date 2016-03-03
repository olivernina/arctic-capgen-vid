import sys
import os
import argparse
import pandas as pd
import re
import numpy as np
import download

def main(argv):
 
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'src_dir',
        help = 'directory where videos are'
    )
    arg_parser.add_argument(
        'dst_dir',
        help = 'directory where to store frames'
    )
    # arg_parser.add_argument(
    #     'list_classes',
    #     help= 'list of classes'
    # )
 
    args = arg_parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    # list_classes = args.list_classes
 
    src_files = os.listdir(src_dir)
 
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
 
 
    # df = pd.read_csv(list_classes,delimiter=' ')
    # dict_classes = df.to_dict()
 
    # classes_allowed = [dict_classes['class_name'][i] for i in range(0,len(dict_classes['class_id']))]
 
 
    for video_file in src_files:
 

        src_path =  os.path.join(src_dir,video_file)

        dst_path = os.path.join(dst_dir, video_file)

        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            command = 'ffmpeg -i '+ src_path+' -s 256x256 '+ dst_path + '/%5d.jpg'
            print command
            os.system(command)



def get_frames(vid_dir,movie_dir,dst_dir,video_file):
    src_dir = os.path.join(vid_dir,movie_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_path =  os.path.join(src_dir,video_file)
    if not os.path.exists(src_path):
        dataset = None
        # download.video_mpii('/media/onina/sea2/datasets/mpii/videos',movie_dir,video_file)
        if dataset=='mvad':
            download.video(vid_dir,movie_dir,video_file)

    dst_path = os.path.join(dst_dir, video_file)

    if not os.path.isdir(dst_path) or len(os.listdir(dst_path))==0:
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        command = 'ffmpeg -i '+ src_path+' -s 256x256 '+ dst_path + '/%5d.jpg'
        print command
        os.system(command)
    else:
        print('frames already extracted')
    return dst_path

def get_frames_mpii(vid_dir,movie_dir,dst_dir,video_file):
    src_dir = os.path.join(vid_dir,movie_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_path =  os.path.join(src_dir,video_file)
    dst_path = None
    if not os.path.exists(src_path):
        dataset = None
        src_path = os.path.join('/media/onina/sea2/datasets/mpii/videos', movie_dir,video_file)
        if not os.path.exists(src_path):
            download.video_mpii('/media/onina/sea2/datasets/mpii/videos',movie_dir,video_file)
        else:
            print 'already downloaded ...'
        if dataset=='mvad':
            download.video(vid_dir,movie_dir,video_file)
        # return None
        dst_path = os.path.join('/media/onina/sea2/datasets/mpii/frames', video_file)

    # dst_path = os.path.join(dst_dir, video_file)

        if not os.path.isdir(dst_path) or len(os.listdir(dst_path))==0:
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            command = 'ffmpeg -i '+ src_path+' -s 256x256 '+ dst_path + '/%5d.jpg'
            print command
            os.system(command)
        else:
            print('frames already extracted')
    return dst_path



def sample_frames(src_dir,video_name,num_samples,rate=-1): #TODO: currently it samples number of frames with same distance, could be done more consecutive ones
#Venugopalan sample every 10 frames
#http://arxiv.org/pdf/1412.4729v3.pdf

    frames = os.listdir(os.path.join(src_dir,video_name))
    frames.sort()

    if num_samples >1:
        if rate <0:
            if len(frames)< num_samples: #this will cause rate to be zero if the clip is short
                a = np.array(frames).repeat(num_samples/len(frames)+1)
                frames = a.tolist()

            rate = len(frames)/num_samples
            if rate < 1:
                print 'rate: '+rate

        frames_sampled = list()
        counter = 1
        for i in range(0,len(frames),rate):
            frames_sampled.append(frames[i])
            counter = counter+1
            if counter > num_samples:
                break
    elif num_samples ==1:
        mid_idx = len(frames)/2
        frames_sampled = [frames[mid_idx]]

    return frames_sampled
 
if __name__=='__main__':
    main(sys.argv)