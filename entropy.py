import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

import dataset


dirpath = '/home/glg/projets/ift6266_projet/scratch/30029.hades'
entrs = np.zeros(1000)


last = None
entrs_a = np.zeros(1000)
for i, path in enumerate([dirpath + '/test_image_{}.jpg'.format(i) for i in range(1000)]):
    img = dataset.load_image(path)
    border, patch = dataset.get_flattened_example(img)

    pb, _ = np.histogram(border, bins=range(2**8), density=True)
    pp, _ = np.histogram(patch, bins=range(2**8), density=True)

    # print sp.entropy(pb), sp.entropy(pp), sp.entropy(pb, pp), sp.entropy(pp, pb)

    entrs[i] = sp.entropy(pp, pb)

    if last is None:
        last = entrs[i]

    alpha = 0.99
    entrs_a[i] = alpha * last + (1 - alpha) * entrs[i]
    last = entrs_a[i]
    
    
plt.plot(np.arange(1000.0), entrs, 'o')
plt.plot(entrs_a)
plt.show()
