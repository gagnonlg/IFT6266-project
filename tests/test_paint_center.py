import numpy as np
import theano.tensor as T
import network

net = network.Network(paint_center=True)
net.add(network.Convolution(3, 3, 33, 33, border_mode='valid'))
net.compile(
    batch_size=1,
    cache_size=(1, (3, 64, 64), 1),
    vartype=(T.tensor4, T.matrix)
)

testi = np.random.uniform(size=(1, 3, 64, 64))
result = net(testi)
np.testing.assert_allclose(result[:,:,0:16,0:48], testi[:,:,0:16,0:48])
np.testing.assert_allclose(result[:,:,16:64,0:16], testi[:,:,16:64,0:16])
np.testing.assert_allclose(result[:,:,48:64,16:64], testi[:,:,48:64,16:64])
np.testing.assert_allclose(result[:,:,0:48,48:64], testi[:,:,0:48,48:64])

net.save('TEST.h5')
net2 = network.Network.load('TEST.h5')
result2 = net2(testi)
np.testing.assert_allclose(result[:,:,0:16,0:48], result2[:,:,0:16,0:48])
np.testing.assert_allclose(result[:,:,16:64,0:16], result2[:,:,16:64,0:16])
np.testing.assert_allclose(result[:,:,48:64,16:64], result2[:,:,48:64,16:64])
np.testing.assert_allclose(result[:,:,0:48,48:64], result2[:,:,0:48,48:64])
