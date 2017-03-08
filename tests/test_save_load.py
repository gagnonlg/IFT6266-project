import theano
import network

mlp = network.Network()
mlp.add(network.Dropout(0.2))
mlp.compile(lr=0.01, momentum=0.5, batch_size=256, cache_size=(1000,1,1))

mlp.save('test_model_2.gz')
mlp = network.Network.load('test_model_2.gz')
