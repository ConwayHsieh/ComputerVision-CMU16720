import numpy as np
import torchvision
import util
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import visual_words
import visual_recog
import skimage.io
import pdb
#import pprint
#pp = pprint.PrettyPrinter(indent=4)

#debugging
#import pdb
#pdb.set_trace()

if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	#path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	#path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
	#path_img = "../data/aquarium/sun_aairflxfskjrkepm.jpg"
	#path_img = "../data/aquarium/sun_aavhedlomnpnwhbf.jpg"
	#path_img = "../data/aquarium/sun_akjefkqgnlztllzo.jpg"

	#image = skimage.io.imread(path_img)
	#image = image.astype('float')/255
	#filter_responses = visual_words.extract_filter_responses(image)
    
    #util.display_filter_responses(filter_responses)
	#visual_words.compute_dictionary(num_workers=num_cores)
    
	#dictionary = np.load('dictionary.npy')
	#wordmap = visual_words.get_visual_words(image,dictionary)
	#print(wordmap)
	#filename = "vis1"
	#util.save_wordmap(wordmap, filename)
	#plt.show()
	
	#dict_size = dictionary.shape[0]

	#print(np.max(wordmap))
	#visual_recog.get_feature_from_wordmap(wordmap,dict_size)
	#visual_recog.get_feature_from_wordmap_SPM(wordmap,3,dict_size)
	#visual_recog.build_recognition_system(num_workers=num_cores)

	conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	print(conf)
	print(np.diag(conf).sum()/conf.sum())

	pass