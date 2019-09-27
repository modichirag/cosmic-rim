import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import flowpm
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

sys.path.append('./utils')
import tools

bs, nc = 100, 32
nsteps = 5
ainit = 0.1

stages = np.linspace(ainit, 1.0, nsteps, endpoint=True)
pk = np.loadtxt('../data/ics_matterpow_0.dat')
ipklin = iuspline(pk[:, 0], pk[:, 1])

print('loaded')

for seed in range(100, 10000, 100):

    print(seed)
##    initial_conditions = flowpm.linear_field(nc,          # size of the cube
##                                             bs,         # Physical size of the cube
##                                             ipklin,      # Initial powerspectrum
##                                             seed=seed,
##                                             batch_size=1)
##
    ick = tools.readbigfile('../data/L%d-N%d-B1-T%d/S%d/linear/LinearDensityK/'%(bs, nc, nsteps, seed))
    ic = np.fft.irfftn(ick)*nc**3
    initial_conditions = tf.cast(tf.expand_dims(tf.constant(ic), 0), tf.float32) - 1.
    
    print(initial_conditions)
    
    # Sample particles
    state = flowpm.lpt_init(initial_conditions, a0=ainit)   

    # Evolve particles down to z=0
    final_state = flowpm.nbody(state, stages, nc)         

    # Retrieve final density field
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])


    with tf.Session() as sess:
        ic, sim = sess.run([initial_conditions, final_field])
        #sim = sess.run(final_field)

    print(ic.mean())
    np.save('../data/L%d-N%d-B1-T%d/S%d/fpm-s'%(bs, nc, nsteps, seed), np.squeeze(ic))
    np.save('../data/L%d-N%d-B1-T%d/S%d/fpm-d'%(bs, nc, nsteps, seed), np.squeeze(sim))

