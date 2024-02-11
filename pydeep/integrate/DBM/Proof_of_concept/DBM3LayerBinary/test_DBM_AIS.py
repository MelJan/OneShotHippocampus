import sys

sys.path.append("/home/melchj57/workspacePy/PyDeep/")
import os

os.nice(19)

import numpy as numx
import pydeep.base.numpyextension as npExt
import DBM3LayerBinary.model as MODEL
import DBM3LayerBinary.trainer as TRAINER
import pydeep.rbm.model as RBM_MODEL
import pydeep.rbm.trainer as RBM_TRAINER
import DBM3LayerBinary.estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
import pydeep.misc.toyproblems as PROBLEMS
import mkl


def train_trial(trial_start,
                trial_end,
                data,
                folder,
                offset_typ,
                epochs=500,
                M=500,
                O=500,
                method='PCD',
                lr=0.001,
                k=[1, 1],
                meanfield=0.0001,
                batch_size=100):
    epsilon = lr * numx.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    if offset_typ[0] == 'D' or offset_typ[0] == '0':
        epsilon[5] *= 0.0
    if offset_typ[1] == '0':
        epsilon[6] *= 0.0
    if offset_typ[2] == '0':
        epsilon[7] *= 0.0

    N = data.shape[1]
    num_batches = data.shape[0] / batch_size
    num_trails = trial_end - trial_start + 1

    measurer = MEASURE.Stopwatch()
    for trial in xrange(trial_start, trial_end + 1):

        numx.random.seed(42 + trial)

        dbm = MODEL.BinaryBinaryDBM(N, M, O, offset_typ, data)
        if method == 'PCD':
            trainer = TRAINER.PCD(dbm, batch_size)
        if method == 'CD':
            trainer = TRAINER.CD(dbm, batch_size)
        if method == 'PT-20':
            trainer = TRAINER.PT(dbm, batch_size, 20)

        # Start time measure and training
        for epoch in xrange(0, epochs + 1):
            for b in xrange(0, data.shape[0], batch_size):
                trainer.train(data=data[b:b + batch_size, :],
                              epsilon=epsilon,
                              k=k,
                              offset_typ=offset_typ,
                              meanfield=meanfield)
            print '\r', trial, "-", epoch, "-", measurer.get_expected_end_time(
                ((trial - trial_start) * epochs) + epoch + 1, num_trails * epochs),
            if epoch > 0 and (epoch % 100 == 0):
                IO.save_object(dbm,
                               folder + offset_typ + "_" + str(trial) + "_" + str(epoch) + "_" + str(N) + "x" + str(
                                   M) + "x" + str(O) + "_" + str(method) + ".dbm", False)

        IO.save_object(dbm, folder + offset_typ + "_" + str(trial) + "_" + str(epochs) + "_" + str(N) + "x" + str(
            M) + "x" + str(O) + "_" + str(method) + ".dbm", False)
    measurer.end()
    print "Done"
    print measurer.get_start_time(), ' -- ', measurer.get_end_time(), ' : ', measurer.get_interval()


mkl.set_num_threads(2)

# Load Data
train_set, _, valid_set, _, _, _ = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz", True)
data = numx.vstack((train_set, valid_set))
print 'MNIST'
train_trial(trial_start=numx.int32(sys.argv[2]),
            trial_end=numx.int32(sys.argv[3]),
            data=data,
            folder="MNIST/",
            offset_typ=str(sys.argv[1]),
            epochs=1000,
            M=500,
            O=500,
            method='PCD',
            lr=0.001,
            k=[1, 1],
            meanfield=0.0001,
            batch_size=100)