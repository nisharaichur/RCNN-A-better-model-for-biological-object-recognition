import sys
sys.path.append('../')
from os.path import abspath
import matplotlib.pyplot as plt
from digitclutter import generate, io
from scipy.io import savemat, loadmat

if __name__=='__main__':
	images = loadmat('test/test.mat')
	debris_array = generate.make_debris(images['images'].shape[0], n_debris = [10, 11])
	images_with_debris = generate.add_debris(images['images'], debris_array)
	savemat(abspath('Debris_10/train_images_with_debris_10.mat'), {'images': images_with_debris, 'targets': images['targets']})

