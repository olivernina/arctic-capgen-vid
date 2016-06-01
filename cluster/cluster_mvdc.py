__author__ = 'onina'

import os
import sys
from time import gmtime, strftime

def main(argv):
    os.system('rm *.pbs')

    s = int(argv[1]) #0
    e = int(argv[2]) #8000
    step =int(argv[3]) #200
    idx = 0
    local_time = strftime("%H:%M:%S",gmtime())
    for i in range(s,e,step):
        start = i
        end = i + step
        job_name  = 'features'+str(idx)+'.pbs'
        f = open(job_name,'w')
        f.write("#!/bin/bash \n")
        f.write("#PBS -A AFSNW35489ANO\n")
        f.write("#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1\n")
        f.write("#PBS -q GPU\n")
        f.write("#PBS -l walltime=0:30:00\n")
        f.write("#PBS -N feats_"+str(start)+"_"+str(end)+"\n")
        f.write("#PBS -j oe\n")
        f.write("module load cuda/7.5\n")
        f.write("module load anaconda/2.3.0\n")
        f.write("module load caffe/20160219\n")
        f.write("cd /p/home/oliver/nips/viddesc/arctic-capgen-vid\n")
        f.write("python process_features2.py /p/work2/projects/ryat/datasets/vid-desc/mvdc/frames/ /p/work2/projects/ryat/datasets/vid-desc/mvdc/features/ '.avi' "+str(start)+" "+str(end))
        f.close()
        command = 'qsub '+job_name
        print command
        os.system(command)
        idx+=1





if __name__=="__main__":


   if len(sys.argv)<2:
       print "Need more arguments \ncluster_jobs.py start end step"
   else:
       main(sys.argv)
