import numpy as np
import matplotlib.pyplot as plt
#
import sys, os
sys.path.append('./utils/')
import tools
import datatools as dtools
from time import time
from pixelcnn3d import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
 #
import tensorflow as tf
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_hub as hub

import tensorflow.contrib.slim as slim
from layers import wide_resnet, wide_resnet_snorm, valid_resnet
from layers import  wide_resnet_snorm
import tensorflow_probability
import tensorflow_probability as tfp
tfd = tensorflow_probability.distributions
tfd = tfp.distributions
tfb = tfp.bijectors


import models
import logging
from datetime import datetime

#############################
seed_in = 3
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)

bss, ncc = [100, 200], [32, 64]
batch_size = [32, 8]
cube_sizes = np.array(ncc)
nsizes = len(cube_sizes)
bsnclist = list(zip(bss, ncc))
for i in bsnclist:
    print(i)
    
numd = 1e-3
num = [int(numd*bs**3) for bs in bss]
R1 = 3
R2 = 3*1.2
knylist = [np.pi*nc/bs for nc, bs in bsnclist]
kklist = [tools.fftk((nc, nc, nc), bs) for nc, bs in bsnclist]

#############################

stellar = False
distribution = 'normal'
#suff = 'pad0-pix-Hpnn-map8-mix4'
suff = 'pad0-pix-cic-pcicmdncmask4normmix-map8-4normmix'
n_mixture = 4

pad = int(0)

savepath = '../models/n10/%s/'%suff
try : os.makedirs(savepath)
except: pass


fname = open(savepath + 'log', 'w+', 1)
ftname = ['cic']
tgname = ['pad0-cic-pcic-cmask-4normmix-pcic.npy']
nchannels = len(ftname)
ntargets = len(tgname)
modelname = '_mdn_pixmodel_fn'
modelfunc = getattr(models, modelname)
rprob = 0.5

print('Features are : ', ftname, file=fname)
print('Target are : ', tgname, file=fname)
print('Model Name : ', modelname, file=fname)
print('Distribution : ', distribution, file=fname)
print('No. of components : ', n_mixture, file=fname)
print('Pad with : ', pad, file=fname)
print('Rotation probability = %0.2f'%rprob, file=fname)
fname.close()

#############################
##Read data and generate meshes

 
cube_features, cube_target = [[] for i in range(len(cube_sizes))], [[] for i in range(len(cube_sizes))]

def generate_training_data(seed, bs, nc):

    
    j = np.where(cube_sizes == nc)[0][0]
    
    path = '../data/make_data_code/L%d-N%d-B1-T5/S%d/'%(bs, nc, seed)
    #path = '../data/L%d-N%d-B1-T5/S%d/'%(bs, nc, seed)

    try:

        mesh = {}
    #    mesh['s'] = np.load(path + 'fpm-s.npy')
        mesh['cic'] = np.load(path + 'fpm-d.npy')    
    #    mesh['logcic'] = np.log(1 + mesh['cic'])
    #    mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
    #    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
    #    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
    #    mesh['GD'] = mesh['R1'] - mesh['R2']
    #
    #    hpath = '../data/L%d-N%d-B2-T10/S%d/fastpm_1.0000/LL-0.200/'%(bs, nc*4, seed)
        ftlist = [mesh[i].copy() for i in ftname]
        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
        features = [np.stack(ftlistpad, axis=-1)]


        target = np.load(path + tgname[0])

        cube_features[j] = cube_features[j] + features
        cube_target[j] = cube_target[j] + [np.expand_dims(target, -1)]

    except Exception as e: print(e)



#############################





class MDNEstimator(tf.estimator.Estimator):
    """An estimator for distribution estimation using Mixture Density Networks.
    """

    def __init__(self,
                 nchannels,
                 n_y,
                 n_mixture,
                 optimizer=tf.train.AdamOptimizer,
                 dropout=None,
                 model_dir=None,
                 config=None):
        """Initializes a `MDNEstimator` instance.
        """


        def _model_fn(features, labels, mode):
            return modelfunc(features, labels, 
                             nchannels, n_y, n_mixture, dropout,
                             optimizer, mode, pad,
                             cfilter_size=None, f_map=8, distribution=distribution)


        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)




def mapping_function(inds):
    def extract_batch(inds):
        
        isize = np.random.choice(len(cube_sizes), 1, replace=True)[0]
        batch = int(batch_size[isize]) #int(batch_size*32/cube_sizes[isize])
        inds = inds[:batch]
        trainingsize = cube_features[isize].shape[0]
        inds[inds >= trainingsize] =  (inds[inds >= trainingsize])%trainingsize
        
        features = cube_features[isize][inds].astype('float32')
        targets = cube_target[isize][inds].astype('float32')
        
        for i in range(batch):
            nrotations=0
            while (np.random.random() < rprob) & (nrotations < 3):
                nrot, ax0, ax1 = np.random.randint(0, 3), *np.random.permutation((0, 1, 2))[:2]
                features[i] = np.rot90(features[i], nrot, (ax0, ax1))
                targets[i] = np.rot90(targets[i], nrot, (ax0, ax1))
                nrotations +=1
# #             print(isize, i, nrotations, targets[i].shape)
# #         print(inds)
        return features, targets
    
    ft, tg = tf.py_func(extract_batch, [inds],
                        [tf.float32, tf.float32])
    return ft, tg

def training_input_fn():
    """Serving input fn for training data"""

    dataset = tf.data.Dataset.range(len(np.array(cube_features)[0]))
    dataset = dataset.repeat().shuffle(1000).batch(32)
    dataset = dataset.map(mapping_function)
    dataset = dataset.prefetch(16)
    return dataset

def testing_input_fn():
    """Serving input fn for testing data"""
    dataset = tf.data.Dataset.range(len(cube_features))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_function)
    return dataset

        
#############################################################################
###save


def save_module(model, savepath, max_steps):

    print('\nSave module\n')

    features = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    exporter = hub.LatestModuleExporter("tf_hub", tf.estimator.export.build_raw_serving_input_receiver_fn({'features':features, 'labels':labels},
                                                                       default_batch_size=None))
    modpath = exporter.export(model, savepath + 'module', model.latest_checkpoint())
    modpath = modpath.decode("utf-8") 
    check_module(modpath)

    
def check_module(modpath):
    
    print('\nTest module\n')
    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']


    for j in range(nsizes):

        bs, nc = bsnclist[j]

        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())

            vseeds = np.random.choice(test_features[j].shape[0], 10)
            xxm = test_features[j][vseeds]
            yym = test_target[j][vseeds]
            print('xxm, yym shape = ', xxm.shape, yym.shape)
            preds = sess.run(samples, feed_dict={xx:xxm, yy:yym})


        ##############################
        ##Power spectrum
        shape = [nc,nc,nc]
        kk = tools.fftk(shape, bs)
        kmesh = sum(i**2 for i in kk)**0.5
        print(kmesh.shape)
        print(preds.shape, yym.shape)
        fig, axar = plt.subplots(2, 2, figsize = (8, 8))
        ax = axar[0]
        for iseed, seed in enumerate(vseeds):
            predict, hpmeshd = np.squeeze(preds[iseed]), np.squeeze(yym[iseed])
            k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
            k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
            k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
            ##
            ax[0].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
            ax[1].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)
            
        for axis in ax.flatten():
            axis.legend(fontsize=14, ncol=3)
            axis.set_yticks(np.arange(0, 1.2, 0.1))
            axis.grid(which='both')
            axis.set_ylim(0.,1.1)
        ax[0].set_ylabel('Transfer function', fontsize=14)
        ax[1].set_ylabel('Cross correlation', fontsize=14)
        #
        ax = axar[1]
        vmin, vmax = 1, (hpmeshd[:, :, :].sum(axis=0)).max()
        im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[1])
        

        ax[0].set_title('Prediction', fontsize=15)
        ax[1].set_title('Truth', fontsize=15)
        plt.savefig(savepath + '/vpredict%d-%d.png'%(nc, max_steps))
        plt.show()

        plt.figure()
        plt.hist(yym.flatten(), range=(-1, 20), bins=100, label='target', alpha=0.8)
        plt.hist(preds.flatten(),  range=(-1, 20), bins=100, label='prediict', alpha=0.5)
        plt.legend()
        plt.savefig(savepath + '/hist%d-%d.png'%(nc, max_steps))
        plt.show()
        ##

    
    dosampletrue = False
    if max_steps in [100, 5000, 10000, 15000, 20000, 25000, 30000]:
        dosampletrue = True
        csize = 32
    if dosampletrue: sampletrue(modpath)


def sampletrue(modpath):
    print('sampling true')
    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

    #
    j = 0
    bs, nc = bsnclist[j]
    
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())
        
        start = time()
        vseeds = np.random.choice(test_features[j].shape[0], 1)
        xxm = test_features[j][vseeds]
        yym = test_target[j][vseeds]
        print('xxm, yym shape = ', xxm.shape, yym.shape)
        

        sample = np.zeros_like(yym)
        sample2 = sess.run(samples, feed_dict={xx:xxm, yy:yym*0})
        for i in range(yym.shape[1]):
            for j in range(yym.shape[2]):
                for k in range(yym.shape[3]):
                    data_dict = {xx:xxm, yy:sample}
                    next_sample = sess.run(samples, feed_dict=data_dict)
                    sample[:, i, j, k, :] = next_sample[:, i, j, k, :]
                        
        end = time()
        print('Taken : ', end-start)
        
    
    plt.figure(figsize = (12, 4))
    vmin, vmax = None, None
    plt.subplot(131)
    plt.imshow(yym[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Data')
    plt.subplot(132)
    plt.imshow(sample[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Correct sample')
    plt.subplot(133)
    plt.imshow(sample2[0,...,0].sum(axis=0), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Single pass sample')
    plt.savefig(savepath + '/sampletrue_im%d'%max_steps)

    ##
    ii = 0
    k, ph = tools.power(yym[ii,...,], boxsize=bs)
    k, pp1 = tools.power(sample[ii,...,], boxsize=bs)
    k, pp1x = tools.power(sample[ii,...,], yym[ii,...,], boxsize=bs)
    k, pp2 = tools.power(sample2[ii,...,], boxsize=bs)
    k, pp2x = tools.power(sample2[ii,...,], yym[ii,...,], boxsize=bs)
    
    
    plt.figure(figsize = (10, 4))
    plt.subplot(121)
    # plt.plot(k, ph, 'C%d-'%ii)
    plt.plot(k, pp1/ph, label='Correct')
    plt.plot(k, pp2/ph,  label='Single Pass')
    plt.ylim(0, 1.5)
    plt.grid(which='both')
    plt.semilogx()
    plt.legend()
    # plt.loglog()
    
    plt.subplot(122)
    plt.plot(k, pp1x/(pp1*ph)**0.5)
    plt.plot(k, pp2x/(pp2*ph)**0.5)
    plt.ylim(0, 1)
    plt.grid(which='both')
    plt.semilogx()
    plt.savefig(savepath + '/sampletrue_2pt%d'%max_steps)
#


############################################################################
#############---------MAIN---------################

for seed in range(10, 10000, 10):
    for bs, nc in bsnclist:
        try: generate_training_data(seed, bs, nc)
        except Exception as e: print(e)

     
for i in range(cube_sizes.size):
    cube_target[i] = np.stack(cube_target[i],axis=0)
    cube_features[i] = np.stack(cube_features[i],axis=0)
    print(cube_features[i].shape, cube_target[i].shape)


train_features, train_target = [], []
test_features, test_target = [], []
for i in range(cube_sizes.size):
    trainsize = int(cube_target[i].shape[0]*0.8)
    print(trainsize)
    train_features.append(cube_features[i][:trainsize])
    train_target.append(cube_target[i][:trainsize])
    test_features.append(cube_features[i][trainsize:])
    test_target.append(cube_target[i][trainsize:])
    print(i, train_target[-1].shape, test_target[-1].shape)

    
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
try: os.makedirs(savepath + '/logs/')
except: pass
logfile = datetime.now().strftime('logs/tflogfile_%H_%M_%d_%m_%Y.log')
fh = logging.FileHandler(savepath + logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


#for max_steps in [50, 100, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]:
for max_steps in [50, 100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000]:
    print('For max_steps = ', max_steps)
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000)

    model =  MDNEstimator(nchannels=nchannels, n_y=ntargets, n_mixture=n_mixture, dropout=0.95,
                      model_dir=savepath + 'model', config = run_config)

    model.train(training_input_fn, max_steps=max_steps)
    f = open(savepath + 'model/checkpoint')
    lastpoint = int(f.readline().split('-')[-1][:-2])
    f.close()
    if lastpoint > max_steps:
        print('Don"t save')
        print(lastpoint)
    else:
        print("Have to save")
        save_module(model, savepath, max_steps)
