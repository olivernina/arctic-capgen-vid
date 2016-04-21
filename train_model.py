import sys
sys.path.insert(1,'coco-caption')

import numpy
import os, sys, socket
import time

from config import config
from jobman import DD, expand
import common
import model_attention_mod
import model_icoupled
import model_attention
import model_same_word
import model_fcoupled
import model_iLSTM
import model_const
import model_const2
import model_LSTM
import model_lstmdd
import model_nio
import model_ni
import model_clstm
import model_nf
import model_fc
import model_ic
import model_const_w

def set_config(conf, args, add_new_key=False):
    # add_new_key: if conf does not contain the key, creates it
    for key in args:
        if key != 'jobman':
            v = args[key]
            if isinstance(v, DD):
                set_config(conf[key], v)
            else:
                if conf.has_key(key):
                    conf[key] = convert_from_string(v)
                elif add_new_key:
                    # create a new key in conf
                    conf[key] = convert_from_string(v)
                else:
                    raise KeyError(key)

def convert_from_string(x):
    """
    Convert a string that may represent a Python item to its proper data type.
    It consists in running `eval` on x, and if an error occurs, returning the
    string itself.
    """
    try:
        return eval(x, {}, {})
    except Exception:
        return x
    
def train_from_scratch(config, state, channel):    
    # Model options
    save_model_dir = config[config.model].save_model_dir

    if save_model_dir == 'current':
        config[config.model].save_model_dir = './'
        save_model_dir = './'
        # to facilitate the use of cluster for multiple jobs
        save_path = './model_config.pkl'
    else:
        # run locally, save locally
        save_path = save_model_dir + 'model_config.pkl'
    print 'current save dir ',save_model_dir
    common.create_dir_if_not_exist(save_model_dir)

    reload_ = config[config.model].reload_
    if reload_:
        print 'preparing reload'
        save_dir_backup = config[config.model].save_model_dir
        from_dir_backup = config[config.model].from_dir
        # never start retrain in the same folder
        assert save_dir_backup != from_dir_backup
        print 'save dir ',save_dir_backup
        print 'from_dir ',from_dir_backup
        print 'setting current model config with the old one'
        model_config_old = common.load_pkl(from_dir_backup+'/model_config.pkl')
        set_config(config, model_config_old)
        config[config.model].save_model_dir = save_dir_backup
        config[config.model].from_dir = from_dir_backup
        config[config.model].reload_ = True
    if config.erase_history:
        print 'erasing everything in ',save_model_dir
        os.system('rm %s/*'%save_model_dir)

    # for stdout file logging
    #sys.stdout = Unbuffered(sys.stdout, state.save_model_path + 'stdout.log')
    print 'saving model config into %s'%save_path
    common.dump_pkl(config, save_path)
    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])
    model_type = config.model
    print 'Model Type: %s'%model_type
    print 'Host:    %s' % socket.gethostname()
    print 'Command: %s' % ' '.join(sys.argv)
    if config.model == 'attention':
        model_attention.train_from_scratch(state, channel)
    elif config.model == 'attention_mod':
        model_attention_mod.train_from_scratch(state, channel)
    elif config.model == 'same_word':
        model_same_word.train_from_scratch(state, channel)
    elif config.model == 'icoupled':
        model_icoupled.train_from_scratch(state, channel)
    elif config.model == 'fcoupled':
        model_fcoupled.train_from_scratch(state, channel)
    elif config.model == 'iLSTM':
        model_iLSTM.train_from_scratch(state, channel)
    elif config.model == 'const':
        model_const.train_from_scratch(state, channel)
    elif config.model == 'const2':
        model_const2.train_from_scratch(state, channel)
    elif config.model == 'LSTM':
        model_LSTM.train_from_scratch(state, channel)
    elif config.model == 'lstmdd':
        model_lstmdd.train_from_scratch(state, channel)
    elif config.model == 'clstm':
        model_clstm.train_from_scratch(state, channel)
    elif config.model == 'nio':
        model_nio.train_from_scratch(state, channel)
    elif config.model == 'ni':
        model_ni.train_from_scratch(state, channel)
    elif config.model == 'nf':
        model_nf.train_from_scratch(state, channel)
    elif config.model == 'fc':
        model_fc.train_from_scratch(state, channel)
    elif config.model == 'ic':
        model_ic.train_from_scratch(state, channel)
    elif config.model == 'const_w':
        model_const_w.train_from_scratch(state, channel)
    else:
        raise NotImplementedError()
        
    
def main(state, channel=None):
    set_config(config, state)
    train_from_scratch(config, state, channel)


if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)
    
    state = expand(args)    
    sys.exit(main(state))

