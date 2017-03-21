import numpy
import pylab
from PIL import Image

import theano.tensor as T
import network

# open random image of dimensions 639x516
img = Image.open(open('tests/3wolfmoon.jpg'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.
# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)


conv = network.Network()
conv.add(network.Convolution(2, 3, 9, 9))
# conv.add(network.MaxPool((2,2)))
# conv.add(network.ReLU())
conv.compile(
    lr=0.01,
    momentum=0.0,
    batch_size=1,
    cache_size=(1, (3, 639, 516), (2, 9, 9)),
    vartype=T.tensor4
)

filtered_img = conv(img_)



# # plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
