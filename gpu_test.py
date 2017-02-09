#  https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/test.py
#  http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu
#  https://medium.com/@wchccy/install-theano-cuda-8-in-ubuntu-16-04-bdb02773e1ea#.xszbqnysf



from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

print("Testing Theano library...")
vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
