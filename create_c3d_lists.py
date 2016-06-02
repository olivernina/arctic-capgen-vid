__author__ = 'oliver'



import sys
import os.path
import argparse

import numpy as np
from scipy.misc import imread, imresize
import scipy.io

import cPickle as pickle



def get_features(src_dir,dst_dir,video_dir,f_in,f_out):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_path =  os.path.join(src_dir,video_dir)
    if os.path.exists(src_path):
        dst_path = os.path.join(dst_dir, video_dir.split('.')[0])

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

            # base_dir = os.path.dirname(files)
            frames = os.listdir(src_path)
            frames.sort()

            filenames = [os.path.join(src_path,x) for x in frames]

            print 'processing '+ dst_path+' '+str(len(filenames))+' frames'
            count = sample_frames(src_path,dst_path,filenames,f_in,f_out)
            print ' features created'
            return count


        else:
            print('features already extracted')
        return 0
    else:
        print('video: '+src_path+' doesn\'t exist')
        # sys.exit(0)


def sample_frames(src_path,dst_path,filenames,f_in,f_out):

    counter =0
    for i in range(0, len(filenames)-16,16):
        # input_image = caffe.io.load_image(filenames[i])
        frame_num = filenames[i].split('/')[-1].split('.')[0]
        line_input = src_path+" "+str(int(frame_num))+" 0\n"
        line_output = dst_path+"/"+frame_num+"\n"
        f_in.write(line_input)
        f_out.write(line_output)
        counter += 1

    return counter


def main(argv):

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'frames_dir',
        help = 'directory where videos are'
    )
    arg_parser.add_argument(
        'feats_dir',
        help = 'directory where to store frames'
    )
    arg_parser.add_argument(
        'input'
    )
    arg_parser.add_argument(
        'output'
    )
    arg_parser.add_argument(
        'ext',
        help = 'video extension'
    )
    arg_parser.add_argument(
        'start',
        help = 'start video index'
    )
    arg_parser.add_argument(
        'end',
        help = 'end video index'
    )


    args = arg_parser.parse_args()
    frames_dir = args.frames_dir
    feats_dir = args.feats_dir
    ext = args.ext
    start = int(args.start)
    end = int(args.end)
    input_file = args.input
    output_file = args.output

    f_out = open(output_file,'w')
    f_in = open(input_file,'w')


    vid_frames = os.listdir(frames_dir)

    if not os.path.isdir(feats_dir):
        os.mkdir(feats_dir)

    total_count = 0


    for i,files in enumerate(vid_frames[start:end]):

        feat_filename = files.split('/')[-1].split(ext)[0]
        feat_file_path = os.path.join(feats_dir,feat_filename)

        # if os.path.exists(feat_file_path):
        #
        #     print('features already extracted '+feat_file_path)
        #     continue
        # else:
        count = get_features(frames_dir,feats_dir,files.split('/')[-1],f_in,f_out)

        total_count+=count

        print str(i)+'/'+str(len(vid_frames))


    f_out.close()
    f_in.close()
    batch_size = 50
    print "features to process:"+str(total_count)+" num of batches: "+str(total_count/batch_size)



if __name__=='__main__':
    main(sys.argv)
