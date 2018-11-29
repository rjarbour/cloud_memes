from neon.callbacks.callbacks import Callbacks, LossCallback, MetricCallback, SerializeModelCallback
from neon.data import CIFAR10
from neon.initializers import Gaussian
from neon.layers import Affine, Linear, Convolution, Pooling, GeneralizedCost, Dropout, BatchNorm, Activation
from neon.models import Model
from neon.optimizers import Adam
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification, Accuracy
from neon import logger as neon_logger
from neon.backends import gen_backend

be = gen_backend(backend='gpu', batch_size=64)
print(be)

cifar10 = CIFAR10()
train = cifar10.train_iter
test = cifar10.valid_iter

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

# setup model layers
layers = [Convolution((3,3,32), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Convolution((3,3,32), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Pooling(fshape=2, strides=2),
          Dropout(keep=0.2),
          Convolution((3,3,64), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Convolution((3,3,64), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Pooling(fshape=2, strides=2),
          Dropout(keep=0.3),
          Convolution((3,3,128), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Convolution((3,3,128), init=init_norm),
          Activation(Rectlin()),
          BatchNorm(),
          Pooling(fshape=2, strides=2),
          Dropout(keep=0.4),
          Linear(nout=1024, init=init_norm, activation=Rectlin()),
          Affine(nout=10, init=init_norm, activation=Softmax())]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# setup optimizer
optimizer = Adam(learning_rate=10 ** -5, beta_1=0.9, beta_2=0.999)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test)
callbacks.add_callback(MetricCallback(eval_set=train, metric=Accuracy(), epoch_freq=1))
callbacks.add_callback(SerializeModelCallback(save_path="./model.prm"))
callbacks.add_save_best_state_callback("./best_state.pkl")


# run fit
mlp.fit(train, optimizer=optimizer, num_epochs=50, cost=cost, callbacks=callbacks)

error_rate = mlp.eval(test, metric=Misclassification())
neon_logger.display("Train Accuracy - {}".format(100 * mlp.eval(test, metric=Accuracy())))
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))