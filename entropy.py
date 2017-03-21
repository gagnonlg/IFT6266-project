import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import h5py as h5

import dataset


dirpath = '/home/glg/projets/ift6266_projet/scratch/30029.hades'
entrs = np.zeros(1000)
entrs2 = np.zeros(1000)

datapath = '/home/glg/projets/ift6266_projet/IFT6266-project/mlp_dataset.h5'
h5dataset = h5.File(datapath, 'r')
tgt = h5dataset['val/target'][0]


ptgt, _ = np.histogram(tgt, bins=range(2**8), density=True)

last = None
entrs_a = np.zeros(1000)
dists = np.zeros(1000)
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

    d = tgt - patch
    dists[i] = np.dot(d, d)

    entrs2[i] = sp.entropy(pp, ptgt)
    
    
plt.plot(np.arange(1000.0), entrs, 'o', label='Q = border')
plt.plot(np.arange(1000.0), entrs2, 'o', label='Q = target patch')
#plt.plot(entrs_a)
plt.xlabel('epoch')
plt.ylabel('KL(generated | Q)')
plt.legend(loc='best')
#plt.show()
plt.savefig('kl_vs_epoch.png')
plt.close()

plt.plot(np.arange(1000.0), dists, 'o')
plt.xlabel('epoch')
plt.ylabel('||target - generated)||_2^2')
#plt.show()
plt.savefig('se_vs_epoch.png')
