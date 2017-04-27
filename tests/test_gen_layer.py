import numpy as np
import theano.tensor as T
import network

net = network.Network()
net.add(network.Generator())
net.compile(
    batch_size=1,
    cache_size=(1, (3, 64, 64), 1),
    vartype=(T.tensor4, T.matrix)
)

testi = np.random.uniform(size=(1, 3, 64, 64))
result = net(testi)
np.testing.assert_allclose(result[:,1:,:,:], testi)

net.save('TEST.h5')
net2 = network.Network.load('TEST.h5')
result2 = net2(testi)
np.testing.assert_allclose(result[:,1:,:,:], result2[:,1:,:,:])
