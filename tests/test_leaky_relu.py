import numpy as np
import network

net = network.Network()
net.add(network.ReLU(alpha=0.2))

net.compile(batch_size=10, cache_size=(10, 10, 1))

x = -1 * np.random.uniform(size=(10,10))
y = net(x)
np.testing.assert_array_less(y, np.full_like(y, 0.0))
