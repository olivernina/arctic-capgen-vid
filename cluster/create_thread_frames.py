__author__ = 'onina'

import os
import sys
from time import gmtime, strftime

def main(argv):
    os.system('rm thread_frames.sh')

    s = int(argv[1]) #0
    e = int(argv[2]) #8000
    step =int(argv[3]) #200
    idx = 0
    job_name  = 'thread_frames.sh'
    f = open(job_name,'w')
    f.write("#!/bin/bash \n")
    for i in range(s,e,step):
        start = i
        end = i + step
        f.write("python process_features.py /p/work2/projects/ryat/datasets/vid-desc/vtt/videos/ /p/work2/projects/ryat/datasets/vid-desc/vtt/frames/ '.mp4' "+str(start)+" "+str(end)+" &\n")
        idx+=1
    # command = 'qsub '+job_name
    # print command
    # os.system(command)
    f.close()




if __name__=="__main__":


   if len(sys.argv)<2:
       print "Need more arguments \ncreate_thread_frames.py start end step"
   else:
       main(sys.argv)
