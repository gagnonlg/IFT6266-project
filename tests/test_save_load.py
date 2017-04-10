import tempfile
import numpy as np
import network
import theano.tensor as T

np.random.seed(20820)

def gen_data_id():
    dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    dataY = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    v_dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    v_dataY = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    t_dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    return dataX, dataY, v_dataX, v_dataY, t_dataX

def test_layer(layer, gen_data):
    """ Save/load test for layers which do not change size of input """

    print '-> Testing layer ' + layer.__class__.__name__

    dataX, dataY, v_dataX, v_dataY, t_dataX = gen_data()

    xs = len(dataX.shape)
    if xs == 2:
        vartypeX = T.matrix
    elif xs == 3:
        vartypeX = T.tensor3
    elif xs == 4:
        vartypeX = T.tensor4

    ys = len(dataY.shape)
    if ys == 2:
        vartypeY = T.matrix
    elif ys == 3:
        vartypeY = T.tensor3
    elif ys == 4:
        vartypeY = T.tensor4

        
    net = network.Network()
    net.add(layer)
    
    net.compile(
        lr=0.01,
        momentum=0.0,
        batch_size=dataX.shape[0],
        cache_size=(dataX.shape[0], dataX.shape[1:], dataY.shape[1:]),
        vartype=(vartypeX, vartypeY)
    )

    net.train(dataX, dataY, (v_dataX, v_dataY), n_epochs=1)

    
    testY = net(t_dataX)

    with tempfile.NamedTemporaryFile() as tmp:
        net.save(tmp.name)
        net2 = network.Network.load(tmp.name)

    testY_2 = net2(t_dataX)

    np.testing.assert_allclose(testY, testY_2)

test_layer_id = lambda l: test_layer(l, gen_data_id)
test_layer_id(network.Layer())
test_layer_id(network.ReLU())
test_layer_id(network.Tanh())
test_layer_id(network.Sigmoid())
test_layer_id(network.Softmax())
test_layer_id(network.Dropout(0.75))
test_layer_id(network.ScaleOffset(scale=100.0, offset=-750))
test_layer_id(network.Clip(0.1,0.2))
test_layer_id(network.BatchNorm(n_input=2))

def gen_data_lineartrans():
    dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    dataY = np.random.uniform(size=64*10).reshape((64, 10)).astype('float32')
    v_dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    v_dataY = np.random.uniform(size=64*10).reshape((64, 10)).astype('float32')
    t_dataX = np.random.uniform(size=64*2).reshape((64, 2)).astype('float32')
    return dataX, dataY, v_dataX, v_dataY, t_dataX
   
test_layer(network.LinearTransformation((2,10)), gen_data_lineartrans)

def gen_data_flatten():
    dataX = np.random.uniform(size=64*10).reshape((64,2,5)).astype('float32')
    dataY = np.random.uniform(size=64*10).reshape((64,10)).astype('float32')
    v_dataX = np.random.uniform(size=64*10).reshape((64,2,5)).astype('float32')
    v_dataY = np.random.uniform(size=64*10).reshape((64,10)).astype('float32')
    t_dataX = np.random.uniform(size=64*10).reshape((64,2,5)).astype('float32')
    return dataX, dataY, v_dataX, v_dataY, t_dataX

test_layer(network.Flatten(), gen_data_flatten)

def gen_data_maxpool():
    dataX = np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    dataY = np.random.uniform(size=64*4).reshape((64,2,2)).astype('float32')
    v_dataX = np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    v_dataY = np.random.uniform(size=64*4).reshape((64,2,2)).astype('float32')
    t_dataX = np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    return dataX, dataY, v_dataX, v_dataY, t_dataX

test_layer(network.MaxPool((2,2)), gen_data_maxpool)

def gen_data_conv():
    dataX =np.random.uniform(size=64*16).reshape((64,1,4,4)).astype('float32')
    # shape - fshape + 1
    # 4 - 2 + 1 = 3
    dataY = np.random.uniform(size=64*9).reshape((64,1,3,3)).astype('float32')
    vdataX=np.random.uniform(size=64*16).reshape((64,1,4,4)).astype('float32')
    vdataY= np.random.uniform(size=64*9).reshape((64,1,3,3)).astype('float32')
    tdataX=np.random.uniform(size=64*16).reshape((64,1,4,4)).astype('float32')

    return dataX,dataY,vdataX,vdataY,tdataX

test_layer(
    network.Convolution(1, 1, 2, 2, border_mode='valid'),
    gen_data_conv
)

def gen_data_rnn():
    dataX =np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    dataY =np.random.uniform(size=64*5).reshape((64,5)).astype('float32')
    vdataX =np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    vdataY =np.random.uniform(size=64*5).reshape((64,5)).astype('float32')
    tdataX =np.random.uniform(size=64*16).reshape((64,4,4)).astype('float32')
    return dataX,dataY,vdataX,vdataY,tdataX

test_layer(
    network.Recurrent(
        n_feature=4,
        n_state=10,
        n_out=5,
        last_output_only=True
    ),
    gen_data_rnn
)

test_layer(
    network.LSTM(
        n_feature=4,
        n_state=5,
        last_state_only=True
    ),
    gen_data_rnn
)                
