import sys
sys.path.append('../')
from os.path import abspath
import matplotlib.pyplot as plt
from digitclutter import generate, io

if __name__=='__main__':
	n_samples = 500
	clutter_list = []
	for i in range(n_samples):
	    clutter_list += [generate.sample_clutter()]
	clutter_list = io.name_files('test', clutter_list=clutter_list)
	io.save_image_set(clutter_list, 'test/test.csv')
	loaded_clutter_list = io.read_image_set('test/test.csv')
	for cl in clutter_list:
	    cl.render_occlusion()
	fname_list = [cl.fname for cl in clutter_list]
	images_dict = io.save_images_as_mat(abspath('test/test.mat'), clutter_list, (32,32), fname_list=fname_list, overwrite_wdir=True)
	#plt.matshow(images_dict['images'][4,:,:,0], cmap = plt.cm.gray, vmin=0, vmax=255)
	debris_array = generate.make_debris(n_samples, n_debris = [2,3])
	images_with_debris = generate.add_debris(images_dict['images'], debris_array)







