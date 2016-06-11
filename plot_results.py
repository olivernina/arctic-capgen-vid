__author__ = 'oliver'


import matplotlib
matplotlib.use('Agg') # Or any other X11 back-end


import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import colors
import pickle
import sys
import math
from numpy import genfromtxt
import os
import six
from random import shuffle
# First arg is file to de-pickle, second arg is "isTest"
work_dir = ''
xmin = 20000

colors_ = list(six.iteritems(colors.cnames))

# Add the single letter colors.
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

# Transform to hex color values.
hex_ = [color[1] for color in colors_]
# shuffle(hex_)
hex_ = hex_[0:-1:2]
def bleu_plot():

    # pretty_colors = ['#FC474C','#8DE047','#FFDD50','#53A3D7']

    max_x = 0
    max_y = 0

    column_num = -11 #cider_val = -8, blue4_val=11, ..., ROUGE= 10,METEOR=11

    files = os.listdir(work_dir)
    dirs = []
    res_files = [os.path.join(work_dir,file,'train_valid_test.txt') for file in files  if os.path.exists(os.path.join(work_dir,file,'train_valid_test.txt'))]

    for i,filename in enumerate(res_files):
            data = genfromtxt(filename,delimiter=' ')

            init_val = 0
            if i==0:
                fig = pyplot.figure(figsize=(6, 6))
                axes = pyplot.gca()
                pyplot.grid()

                max_y = data[init_val:,column_num].max()
                max_x = data[init_val:,0].max()

                # axes.set_ylim([0, max_y])
                axes.set_ylim([.25,.40])
                # axes.set_xlim([xmin, max_x])
                axes.set_xlim([0, 25])
                pyplot.xlabel('Iterations')
                pyplot.ylabel('BLEU')
                pyplot.title('MSR-VTT')

            # if data[init_val:,column_num].max() > max_y:
            #     max_y = data[init_val:,column_num].max()
            #     axes.set_ylim([0, max_y])
            #
            # if data[init_val:,0].max() > max_x:
            #     max_x = data[init_val:,0].max()
            #     axes.set_xlim([0, max_x])

            pyplot.plot(data[init_val:,0], data[init_val:,column_num], linewidth=2, label=filename.split('/')[-2], color=hex_[i])

    pyplot.legend(loc='upper right', shadow=True, fontsize='medium')
    pyplot.savefig(os.path.join(work_dir,'bleu.eps'))

def cider_plot():

    # pretty_colors = ['#FC474C','#8DE047','#FFDD50','#53A3D7']

    max_x = 0
    max_y = 0

    column_num = -8 #cider_val = -8, blue4_val=11, ..., ROUGE= 10,METEOR=11

    files = os.listdir(work_dir)
    dirs = []
    res_files = [os.path.join(work_dir,file,'train_valid_test.txt') for file in files  if os.path.exists(os.path.join(work_dir,file,'train_valid_test.txt'))]

    for i,filename in enumerate(res_files):
            data = genfromtxt(filename,delimiter=' ')

            init_val = 0
            if i==0:
                fig = pyplot.figure(figsize=(6, 6))
                axes = pyplot.gca()
                pyplot.grid()

                max_y = data[init_val:,column_num].max()
                max_x = data[init_val:,0].max()

                # axes.set_ylim([0, max_y])
                axes.set_ylim([.25,.40])
                # axes.set_xlim([xmin, max_x])
                axes.set_xlim([0, 25])
                pyplot.xlabel('Iterations')
                pyplot.ylabel('CiDER')
                pyplot.title('MSR-VTT')

            # if data[init_val:,column_num].max() > max_y:
            #     max_y = data[init_val:,column_num].max()
            #     axes.set_ylim([0, max_y])
            #
            # if data[init_val:,0].max() > max_x:
            #     max_x = data[init_val:,0].max()
            #     axes.set_xlim([0, max_x])

            pyplot.plot(data[init_val:,0], data[init_val:,column_num], linewidth=2, label=filename.split('/')[-2], color=hex_[i])

    pyplot.legend(loc='upper right', shadow=True, fontsize='medium')
    pyplot.savefig(os.path.join(work_dir,'cider.eps'))

if __name__=="__main__":
    work_dir = sys.argv[1]
    plot_type = sys.argv[2]

    # if plot_type == 'loss': #training loss
    #     loss_plot(sys.argv[3:])
    # if plot_type == 'ppl2': #training ppl2
    #     ppl2_plot(sys.argv[3:])
    # if plot_type == 'val': #val ppl2
    #     val_plot(sys.argv[3:])
    if plot_type == 'bleu':
        bleu_plot()
    if plot_type == 'cider':
        cider_plot()
    # if plot_type == 'rouge':
    #     rouge_plot(sys.argv[3:])
    # if plot_type == 'meteor':
    #     meteor_plot(sys.argv[3:])
    # if plot_type == 'time':
    #     time_plot(sys.argv[3:])