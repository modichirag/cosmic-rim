import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#
import sys, os
sys.path.append('./utils/')
import tools
import datatools as dtools
from time import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
import tensorflow as tf
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_hub as hub


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

print('knylist : ', knylist)


#############################

logoffset = 1e-3
distribution = 'normal'
n_mixture = 4
pad = int(0)
masktype = 'constant'
suff = 'pad%d-cic-mcicnomean-cmask-4normmix'%pad


savepath = '../models/n10/%s/'%suff
try : os.makedirs(savepath)
except: pass


fname = open(savepath + 'log', 'w+', 1)
ftname = ['cic']
tgname = ['mcicnomean']
nchannels = len(ftname)
ntargets = len(tgname)
modelname = '_mdn_mask_model_fn'
modelfunc = getattr(models, modelname)

rprob = 0.5

print('Features are : ', ftname, file=fname)
print('Target are : ', tgname, file=fname)
print('Model Name : ', modelname, file=fname)
print('Distribution : ', distribution, file=fname)
print('Masktype : ', masktype, file=fname)
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
        hpath = '../data/make_data_code/L%d-N%d-B2-T10/S%d/FOF/'%(bs, nc*4, seed)
        hmesh = {}
        hposall = tools.readbigfile(hpath + 'CMPosition/')[1:]    
        massall = tools.readbigfile(hpath + 'Length/')[1:].reshape(-1)
        hposd = hposall[:num[j]].copy()
        massd = massall[:num[j]].copy()
    #    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    #    hmesh['pnn'] = tools.paintnn(hposd, bs, nc)
    #    hmesh['mnn'] = tools.paintnn(hposd, bs, nc, massd)
        hmesh['mcic'] = tools.paintcic(hposd, bs, nc, massd)
        hmesh['mcicnomean'] =  (hmesh['mcic'])/hmesh['mcic'].mean()
    ##    hmesh['mcicovd'] =  (hmesh['mcic'] - hmesh['mcic'].mean())/hmesh['mcic'].mean()
    ##    hmesh['mcicovdR3'] = tools.fingauss(hmesh['mcicovd'], kk, R1, kny)    
    ##    hmesh['pcicovd'] =  (hmesh['pcic'] - hmesh['pcic'].mean())/hmesh['pcic'].mean()
    ##    hmesh['pcicovdR3'] = tools.fingauss(hmesh['pcicovd'], kk, R1, kny)    
    ##    hmesh['lmnn'] = np.log(logoffset + hmesh['mnn'])
    ##
        ftlist = [mesh[i].copy() for i in ftname]
        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
        targetmesh = [hmesh[i].copy() for i in tgname]
        features = [np.stack(ftlistpad, axis=-1)]
        target = [np.stack(targetmesh, axis=-1)]

        cube_features[j] = cube_features[j] + features
        cube_target[j] = cube_target[j] + target

    except Exception as e: print(e)






#############################

class MDNEstimator(tf.estimator.Estimator):
    """An estimator for distribution estimation using Mixture Density Networks.
    """

    def __init__(self,
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
                                             optimizer, mode, pad, distribution=distribution, masktype=masktype)

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
    

#####
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
    train_features.append(cube_features[i][:trainsize])
    train_target.append(cube_target[i][:trainsize])
    test_features.append(cube_features[i][trainsize:])
    test_target.append(cube_target[i][trainsize:])

    print(train_target[-1].shape, test_target[-1].shape)
    

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


for max_steps in [50, 100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000]:
#for max_steps in [100]+list(np.arange(5e3, 7.1e4, 5e3, dtype=int)):
    print('For max_steps = ', max_steps)
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000)

    model =  MDNEstimator(n_y=ntargets, n_mixture=n_mixture, dropout=0.95,
                      model_dir=savepath + 'model', config = run_config)

    model.train(training_input_fn, max_steps=max_steps)
    #save_module(model, savepath, max_steps)
    f = open(savepath + 'model/checkpoint')
    lastpoint = int(f.readline().split('-')[-1][:-2])
    f.close()
    if lastpoint > max_steps:
        print('Don"t save')
        print(lastpoint)
    else:
        print("Have to save")
        save_module(model, savepath, max_steps)



