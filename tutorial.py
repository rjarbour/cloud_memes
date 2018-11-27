from neon.callbacks.callbacks import Callbacks, LossCallback, MetricCallback
from neon.data import CIFAR10
from neon.initializers import Gaussian
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import Adam
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, Accuracy
from neon import logger as neon_logger
from neon.backends import gen_backend

be = gen_backend(backend='cpu', batch_size=128)
print(be)

cifar10 = CIFAR10()
train = cifar10.train_iter
test = cifar10.valid_iter

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers
layers = [Conv(fshape=(5,5,32), init=init_norm, activation=Rectlin()),
          Pooling(fshape=2, strides=1),
          Conv(fshape=(5,5,64), init=init_norm, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Affine(nout=1024, init=init_norm, activation=Rectlin()),
          Affine(nout=10, init=init_norm, activation=Softmax())]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# setup optimizer
optimizer = Adam(learning_rate=10 ** -5, beta_1=0.9, beta_2=0.999)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test)
callbacks.add_callback(LossCallback(eval_set=test, epoch_freq=1))
#callbacks.add_callback(MetricCallback(eval_set=test, metric=Accuracy, epoch_freq=1))

# run fit
mlp.fit(train, optimizer=optimizer, num_epochs=20, cost=cost, callbacks=callbacks)

error_rate = mlp.eval(test, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))