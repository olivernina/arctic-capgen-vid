import cPickle as pkl
import gzip
import os, socket, shutil
import sys, re
import time
from collections import OrderedDict
import numpy
import tables
import theano
import theano.tensor as T
import common
import numpy as np

sys.path.append('skip-thoughts')
import skipthoughts
from scipy import spatial

from multiprocessing import Process, Queue, Manager

hostname = socket.gethostname()
                
class Movie2Caption(object):
            
    def __init__(self, model_type, signature, video_feature,
                 mb_size_train, mb_size_test, maxlen, n_words,dec,
                 n_frames=None, outof=None
                 ):
        self.signature = signature
        self.model_type = model_type
        self.video_feature = video_feature
        self.maxlen = maxlen
        self.n_words = n_words
        self.K = n_frames
        self.OutOf = outof
        self.dec = dec

        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.non_pickable = []

        # self.test_mode = 0 #don't chage this when in production
        self.load_data()


        if dec=='multi-stdist':
            self.st_model = skipthoughts.load_model()
            self.cap_distances = {}

        
    def _filter_googlenet(self, vidID):
        feat = self.FEAT[vidID]
        feat = self.get_sub_frames(feat)
        return feat

    def _load_feat_file(self, vidID):

        # feats_dir =os.path.join(data_dir,'features_chal')
        # feats_dir = '/media/onina/sea2/datasets/lsmdc/features_chal'
        feats_dir = '/p/work2/projects/ryat/datasets/vid-desc/features_chal'
        feat_filename = vidID#files.split('/')[-1].split('.avi')[0]
        feat_file_path = os.path.join(feats_dir,feat_filename)

        if os.path.exists(feat_file_path):
            feat = np.load(feat_file_path)
        else:
<<<<<<< HEAD
            print 'error feature file doesnt exist'+feat_file_path
=======
            print ' error feature file doesnt exist'
>>>>>>> 4e68221e6b459e3d7ee9740a0652f1eabbbb925e

        # feat = self.FEAT[vidID]
        feat = self.get_sub_frames(feat)
        return feat

    def get_video_features(self, vidID):
        if self.video_feature == 'googlenet':
            # y = self._filter_googlenet(vidID)

            if self.signature == 'youtube2text':
                y = self._filter_googlenet(vidID)
            else:
                y = self._load_feat_file(vidID) #this is for large datasets, needs to be fixed with something better
        else:
            raise NotImplementedError()
        return y

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = numpy.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = numpy.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = numpy.array_split(range(n_frames), self.K)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = numpy.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = numpy.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = numpy.concatenate([frames, eos], axis=0)
        return frames
    
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take K of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if self.OutOf:
            raise NotImplementedError('OutOf has to be None')
            frames_ = frames[:self.OutOf]
            if len(frames_) < self.OutOf:
                frames_ = self.pad_frames(frames_, self.OutOf, jpegs)
        else:
            if len(frames) < self.K:
                #frames_ = self.add_end_of_video_frame(frames)
                frames_ = self.pad_frames(frames, self.K, jpegs)
            else:

                frames_ = self.extract_frames_equally_spaced(frames, self.K)
                #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_

    def prepare_data_for_blue(self, whichset):
        # assume one-to-one mapping between ids and features
        feats = []
        feats_mask = []
        if whichset == 'valid':
            ids = self.valid_ids
        elif whichset == 'test':
            ids = self.test_ids
        elif whichset == 'train':
            ids = self.train_ids
        for i, vidID in enumerate(ids):
            feat = self.get_video_features(vidID)
            feats.append(feat)
            feat_mask = self.get_ctx_mask(feat)
            feats_mask.append(feat_mask)
        return feats, feats_mask
    
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
    
        
    def load_data(self):
        if self.signature == 'youtube2text':
            print 'loading youtube2text %s features'%self.video_feature
            dataset_path = common.get_rab_dataset_base_path()+'youtube2text_iccv15/'
            # dataset_path = common.get_rab_dataset_base_path()
            self.train = common.load_pkl(dataset_path + 'train.pkl')
            self.valid = common.load_pkl(dataset_path + 'valid.pkl')
            self.test = common.load_pkl(dataset_path + 'test.pkl')
            self.CAP = common.load_pkl(dataset_path + 'CAP.pkl')
            self.FEAT = common.load_pkl(dataset_path + 'FEAT_key_vidID_value_features.pkl')



            # if self.test_mode:
            #     self.train_ids = ['vid%s'%i for i in range(1,120)]
            #     self.valid_ids = ['vid%s'%i for i in range(120,130)]
            #     self.test_ids = ['vid%s'%i for i in range(130,197)]
            # else:
            self.train_ids = ['vid%s'%i for i in range(1,1201)]
            self.valid_ids = ['vid%s'%i for i in range(1201,1301)]
            self.test_ids = ['vid%s'%i for i in range(1301,1971)]

        elif self.signature == 'lsmdc':
            print 'loading lsmdc %s features'%self.video_feature
            # dataset_path = common.get_rab_dataset_base_path()+'youtube2text_iccv15/'
            dataset_path = common.get_rab_dataset_base_path()
            self.train = common.load_pkl(dataset_path + 'train.pkl')
            self.valid = common.load_pkl(dataset_path + 'valid.pkl')
            self.test = common.load_pkl(dataset_path + 'test.pkl')
            self.CAP = common.load_pkl(dataset_path + 'CAP.pkl')
            self.FEAT = common.load_pkl(dataset_path + 'FEAT_key_vidID_value_features.pkl')

            self.train_ids = self.train
            self.valid_ids = self.valid
            self.test_ids = self.test

        elif self.signature == 'mvad':
            print 'loading mvad %s features'%self.video_feature
            dataset_path = common.get_rab_dataset_base_path()+'mvad/'
            # dataset_path = common.get_rab_dataset_base_path()
            self.train = common.load_pkl(dataset_path + 'train.pkl')
            self.valid = common.load_pkl(dataset_path + 'valid.pkl')
            self.test = common.load_pkl(dataset_path + 'test.pkl')
            self.CAP = common.load_pkl(dataset_path + 'CAP.pkl')
            # self.FEAT = common.load_pkl(dataset_path + 'FEAT_key_vidID_value_features.pkl')


            self.train_ids = self.train
            self.valid_ids = self.valid
            self.test_ids = self.test

        elif self.signature == 'ysvd':
            print 'loading ysvd %s features'%self.video_feature
            dataset_path = common.get_rab_dataset_base_path()+'ysvd/'

            self.all = common.load_pkl(dataset_path + 'all_vids.pkl')
            self.CAP = common.load_pkl(dataset_path + 'CAP.pkl')
            self.FEAT = common.load_pkl(dataset_path + 'FEAT_key_vidID_value_features.pkl')

            print self.FEAT.keys()

            self.train = self.all[0:500]
            self.valid = self.all[501:750]
            self.test = self.all[751:1000]

            self.train_ids = self.train
            self.valid_ids = self.valid
            self.test_ids = self.test


        else:
            raise NotImplementedError()
                
        self.worddict = common.load_pkl(dataset_path + 'worddict.pkl')
        self.word_idict = dict()
        # wordict start with index 2
        for kk, vv in self.worddict.iteritems():
            self.word_idict[vv] = kk
        self.word_idict[0] = '<eos>'
        self.word_idict[1] = 'UNK'
        
        if self.video_feature == 'googlenet':
            self.ctx_dim = 1024
        else:
            raise NotImplementedError()
        self.kf_train = common.generate_minibatch_idx(
            len(self.train), self.mb_size_train)
        self.kf_valid = common.generate_minibatch_idx(
            len(self.valid), self.mb_size_test)
        self.kf_test = common.generate_minibatch_idx(
            len(self.test), self.mb_size_test)
        
def prepare_data(engine, IDs):
    seqs = []
    z_seqs = []
    feat_list = []

    def get_words(vidID, capID):
        rval = None
        if engine.signature == 'youtube2text':
            caps = engine.CAP[vidID]
            for cap in caps:
                if cap['cap_id'] == capID:
                    rval = cap['tokenized'].split(' ')
                    break
        elif engine.signature == 'lsmdc':
            cap = engine.CAP[vidID][0]

            # for cap in caps:
            #     if cap['cap_id'] == capID:
            #         rval = cap['tokenized'].split(' ')
            #         break

            # cap = caps[0]
            # if cap['cap_id'] == capID:
            caption = cap['tokenized']
            rval = cap['tokenized'].split()
        elif engine.signature == 'mvad':
            cap = engine.CAP[vidID][0]
            caption = cap['tokenized']
            rval = cap['tokenized'].split()
        elif engine.signature == 'ysvd':
            cap = engine.CAP[vidID][0]
            caption = cap['tokenized']
            rval = cap['tokenized'].split()

        assert rval is not None
        return rval

    def get_z_seq():
        caps = engine.CAP[vidID]
        num_caps = len(caps)

        if engine.dec == 'multi-random':
            import random
            r = range(1,int(capID)) + range(int(capID)+1,num_caps)
            rand_cap = random.choice(r)
            z_words = get_words(vidID, str(rand_cap))
            z_seq = [engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in z_words]

        elif engine.dec == 'multi-stdist': #'stdist'

            # common.dump_pkl(caps,'/media/onina/SSD/projects/skip-thoughts/caps')

            if not engine.cap_distances.has_key(vidID):

                captions = ["" for x in range(num_caps)]
                for i in range(0,num_caps):
                    cap = caps[i]
                    id = int(cap['cap_id'])
                    caption = cap['caption']
                    udata=caption.decode("utf-8")
                    captions[id] = udata.encode("ascii","ignore")
                    # print captions[id]
                    if captions[id].isspace():
                        captions[id] = captions[0]

                common.dump_pkl(captions,'captions')
                vectors = skipthoughts.encode(engine.st_model,captions)
                caps_dist = spatial.distance.cdist(vectors, vectors, 'cosine')
                engine.cap_distances[vidID] = caps_dist

            caps_dist = engine.cap_distances[vidID]
            query_id = int(capID)
            js =range(0, query_id) + range(query_id+1,num_caps)
            most_distant = np.argmax(caps_dist[query_id,js])

            z_words = get_words(vidID, str(most_distant))
            z_seq = [engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in z_words]

        return z_seq

    def clean_sequences(seqs,z_seqs,feat_list):

        if  engine.dec=="standard":

            lengths = [len(s) for s in seqs]
            if engine.maxlen != None:
                new_seqs = []
                new_feat_list = []
                new_lengths = []
                new_caps = []
                for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
                    # sequences that have length >= maxlen will be thrown away
                    if l < engine.maxlen:
                        new_seqs.append(s)
                        new_feat_list.append(y)
                        new_lengths.append(l)
                        new_caps.append(c)
                lengths = new_lengths
                feat_list = new_feat_list
                seqs = new_seqs

            return seqs,None,feat_list,lengths

        else:
            lengths = [len(s) for s in seqs]
            z_lengths = [len(s) for s in z_seqs]
            if engine.maxlen != None:
                new_seqs = []
                new_zseqs = []
                new_feat_list = []
                new_lengths = []
                new_caps = []
                new_zlengths = []
                for l,z_l, s, y, c in zip(lengths,z_lengths, seqs, feat_list, IDs):
                    # sequences that have length >= maxlen will be thrown away
                    if l < engine.maxlen and z_l < engine.maxlen :
                        new_seqs.append(s)
                        new_zseqs.append(s)
                        new_feat_list.append(y)
                        new_lengths.append(l)
                        new_caps.append(c)
                lengths = new_lengths
                feat_list = new_feat_list
                seqs = new_seqs
                z_seqs = new_zseqs

            return seqs,z_seqs,feat_list,lengths


    for i, ID in enumerate(IDs):
        #print 'processed %d/%d caps'%(i,len(IDs))
        # print ID
        if engine.signature == 'youtube2text':
            # load GNet feature
            vidID, capID = ID.split('_')
        elif engine.signature == 'lsmdc':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 1
        elif engine.signature == 'mvad':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 1
        elif engine.signature == 'ysvd':
            # t = ID.split('_')
            # vidID = '_'.join(t[:-1])
            # capID = t[-1]
            vidID = ID
            capID = 1
        else:
            raise NotImplementedError()
        
        feat = engine.get_video_features(vidID)
        feat_list.append(feat)
        words = get_words(vidID, capID)
        # print words
        seqs.append([engine.worddict[w] if engine.worddict[w] < engine.n_words else 1 for w in words])

        # print engine.dec
        if engine.dec != "standard":
            z_seq = get_z_seq()
            z_seqs.append(z_seq)


    seqs,z_seqs,feat_list,lengths = clean_sequences(seqs,z_seqs,feat_list)

    if len(lengths) < 1:
        return None, None, None, None


    y = numpy.asarray(feat_list)
    # print len(y[1,1])
    y_mask = engine.get_ctx_mask(y)

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    if engine.dec=="standard":
        return x, x_mask, y, y_mask
    else:
        z = numpy.zeros((maxlen, n_samples)).astype('int64')  #This is the other label
        z_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(z_seqs):
            z[:lengths[idx],idx] = s
            z_mask[:lengths[idx]+1,idx] = 1.

        return x, x_mask, y, y_mask,z,z_mask
    
def test_data_engine():
    from sklearn.cross_validation import KFold
    video_feature = 'googlenet' 
    out_of = None
    maxlen = 100
    mb_size_train = 64
    mb_size_test = 128
    maxlen = 50
    n_words = 30000 # 25770 
    signature = 'youtube2text' #'youtube2text'
    engine = Movie2Caption('attention', signature, video_feature,
                           mb_size_train, mb_size_test, maxlen,
                           n_words,
                           n_frames=26,
                           outof=out_of)
    i = 0
    t = time.time()
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids)
        print 'seen %d minibatches, used time %.2f '%(i,time.time()-t0)
        if i == 10:
            break
            
    print 'used time %.2f'%(time.time()-t)
if __name__ == '__main__':
    test_data_engine()


