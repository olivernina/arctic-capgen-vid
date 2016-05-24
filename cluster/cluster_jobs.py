__author__ = 'onina'

import os

os.system('rm *.pbs')

start = 0
end = 8000
step =200
idx = 0
for i in range(start,end,step):
    start = i
    end = i + step
    job_name  = 'features'+str(idx)+'.pbs'
    f = open(job_name,'w')
    f.write("#!/bin/bash \n")
    f.write("#PBS -A AFSNW35489ANO\n")
    f.write("#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1\n")
    f.write("#PBS -q GPU\n")
    f.write("#PBS -l walltime=1:00:00\n")
    f.write("#PBS -N features\n")
    f.write("#PBS -j oe\n")
    f.write("module load cuda/7.5\n")
    f.write("module load anaconda/2.3.0\n")
    f.write("module load caffe/20160219\n")
    f.write("cd /p/home/oliver/nips/viddesc/arctic-capgen-vid\n")
    f.write("python process_features.py /p/work2/projects/ryat/datasets/vid-desc/vtt/frames/ /p/work2/projects/ryat/datasets/vid-desc/vtt/features/ '.mp4' "+str(start)+" "+str(end))
    f.close()
    command = 'qsub '+job_name
    print command
    os.system(command)
    idx+=1