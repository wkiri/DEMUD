#!/user/bin/env python
# Extract feature vectors from all layers of CaffeNet
# and save as CSV
# Sources:
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
# https://github.com/BVLC/caffe/tree/master/examples/feature_extraction
# ~/classify_imagenet.py
#
# Kiri Wagstaff, Jake Lee
# v0.1 - 6/28/17
# v0.2 - 6/29/17
# v0.3 - 7/29/17
# v0.4 - 6/26/18

import sys, os
import numpy as np
# Must do this before importing caffe for it to take effect
os.environ['GLOG_minloglevel'] = '2'	# suppress caffe log output
import caffe
import imghdr
import warnings
import ConfigParser
import csv

caffe.set_mode_cpu()

localdir = os.path.dirname(os.path.abspath(__file__))
print localdir

# print usage message
def usage():	
	print "usage: python feat_csv.py config_file.cfg"
	sys.exit(1)

# main function for exportcsv
def export_csv(model_def, model_weights, mean_image, synset_words,
		layer_list, class_list, qty_list, imageset_dir):
	
	# check number of classes and number of images
	dircount = 0
	imagecount = 0
	
	# Make sure defined classes exist
	# If class_list is empty, use everything
	if class_list == ['']:
		class_list = os.listdir(os.path.join(localdir, imageset_dir))
	for a_dir in os.listdir(os.path.join(localdir, imageset_dir)):
		if ('.' not in a_dir) and (a_dir in class_list):
			dircount = dircount + 1
	if dircount != len(class_list):
		print 'error, some class directories do not exist'
		sys.exit(1)
	

	# Make sure defined images exist
	allflag = False
	for i in range(len(class_list)):
		imagecount = len(os.listdir(os.path.join(localdir, imageset_dir,
			class_list[i])))
		if qty_list == ['']:
			qty_list = [0] * len(class_list)
			qty_list[i] = imagecount
			allflag = True
		if allflag:
			qty_list[i] = imagecount
		if imagecount < int(qty_list[i]):
			print 'error, class %s does not have enough images' % class_list[i]
			sys.exit(1)
	
	# get list of images
	# image_list[class#][image#]
	image_list = []
	label_list = []
	for i in range(len(class_list)):
		temp_list = []
		temp_label_list = []
		for j in range(int(qty_list[i])):
			temp_list.append(os.path.join(localdir, imageset_dir, 
				class_list[i], os.listdir(os.path.join(localdir, 
				imageset_dir, class_list[i]))[j]))
			temp_label_list.append(os.listdir(os.path.join(localdir,
				imageset_dir, class_list[i]))[j])
		image_list.append(temp_list)
		label_list.append(temp_label_list)
	label_list = [an_item for a_list in label_list for an_item in a_list]
	label_list = label_list[::-1]

	# CaffeNet definitions
	net = caffe.Net(model_def, model_weights, caffe.TEST)
	mu = np.load(mean_image)
	mu = mu.mean(1).mean(1)

	# Transform from RGB to BGR
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', mu)            
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	# Reshape to match expected dimensions
	n_images = 1
	n_channels = 3 #BGR
	net.blobs['data'].reshape(n_images, n_channels, 227, 227)
	
	feat_list = []
	# feat_list[class#][image#][layer#]
	for i in range(len(class_list)):
		temp_image_list = []
		for j in range(int(qty_list[i])):
			temp_feat_list = []
			image_filename = image_list[i][j]
			print "\rProcessing ", image_filename, "...",

			if not os.path.exists(image_filename):
				print 'Could not load %s"' % image_filename
				sys.exit(1)

			# Load image
			image = caffe.io.load_image(image_filename)
			transformed_image = transformer.preprocess('data', image)
			net.blobs['data'].data[...] = transformed_image

			# Classify
			output = net.forward()

			feat = np.zeros(1)
			for a_layer in layer_list:
				try:
					feat = net.blobs[a_layer].data[0].copy()
				except KeyError:
					print '\nerror, invalid layer %s' % a_layer	
					layer_list.remove(a_layer)
					continue
				temp_feat_list.append(feat)
			temp_image_list.append(temp_feat_list)

		feat_list.append(temp_image_list)
		# feat_list[class#][image#][layer#]

	k_feat_list = []
	for i in range(0, dircount):
		k_feat_list = k_feat_list + feat_list[i]
	k_feat_list = np.rot90(k_feat_list,3)	
	# k_feat_list[layer#][image#], no class distinction 
	
	layercount = len(k_feat_list)
	print "\nthere are ", layercount, " layers"
	
	for i in range(len(k_feat_list)):
		print "exporting %s.csv" % layer_list[i]
		temp_array = np.vstack(k_feat_list[i])
		temp_list = []
		for j in range(temp_array.shape[0]):
			new_row = temp_array[j].tolist()
			new_row.insert(0, label_list[j])
			temp_list.insert(0, new_row)
			# add string ID
			# add row to temp_list
		
		#export list as csv
		if not os.path.exists('feats'):
			os.makedirs('feats')
		with open("feats/"+imageset_dir+layer_list[i]+".csv",'wb') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
			#print "writing"
			wr.writerows(temp_list)
			#print "done"

	return 1

if __name__ == "__main__":	
	# check args
	if len(sys.argv) != 2:
		usage()
	elif len(sys.argv) == 2 and sys.argv[1] == '-h':
		usage()	
	configfile = sys.argv[1]
	
	# read in config file
	if not os.path.exists(configfile):
		print "could not find config file", configfile
		usage()
	
	config = ConfigParser.ConfigParser()
	config.read(configfile)
	
	imageset_dir = config.get('Files', 'imageset_dir')
	model_dir = config.get('Files', 'model_dir')

	model_def = os.path.join(model_dir, config.get('Files', 'model_def'))
	if not os.path.exists(model_def):
		print 'Could not find model def file %s.' % model_def
		usage()

	model_weights = os.path.join(model_dir, config.get('Files', 'model_weights'))
	if not os.path.exists(model_weights):
		print 'Could not find model weights file %s.' % model_weights
		usage()
	
	mean_image = os.path.join(model_dir, config.get('Files', 'mean_image'))
	if not os.path.exists(mean_image):
		print 'Could not find mean image file %s.' % mean_image
		usage()

	synset_words = os.path.join(model_dir, config.get('Files', 'synset_words'))
	if not os.path.exists(synset_words):
		print 'Could not find synset words file %s.' % synset_words
		usage()

	# get layers and layer counts
	layer_list = config.get('Params', 'layer_list')
	layer_list = layer_list.split(',')

	class_list = config.get('Params', 'class_list')
	class_list = class_list.split(',')
	
	qty_list = config.get('Params', 'qty_list')
	qty_list = qty_list.split(',')
	
	export_csv(model_def, model_weights, mean_image, synset_words, 
		layer_list, class_list, qty_list, imageset_dir)




