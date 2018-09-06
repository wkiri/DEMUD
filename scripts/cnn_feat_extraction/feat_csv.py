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
# v1.0 - 8/30/18

import sys, os
import numpy as np
# Must do this before importing caffe for it to take effect
os.environ['GLOG_minloglevel'] = '2'	# suppress caffe log output
import caffe
import imghdr
import warnings
import ConfigParser
import csv
import copy


localdir = os.path.dirname(os.path.abspath(__file__))

# print usage message
def usage():	
	print "usage: python feat_csv.py config_file.cfg"
	sys.exit(1)

# config file error
def bad_config():
	print "fatal: an error in the config file was encountered."
	sys.exit(1)

# list only files, no directory
def list_files(directory):
	dir_list = os.listdir(directory)
	for afile in dir_list:
		if not os.path.isfile(os.path.join(directory,afile)):
			dir_list.remove(afile)
		if afile[0] == '.':
			# we don't mess with weird files.
			dir_list.remove(afile)
	return dir_list

# main function for exportcsv
def export_csv(model_def, model_weights, mean_image, 
		layer_list, class_list, qty_list, imageset_dir, batch_size):

	# Flag to indicate all classes were specified
	all_flag = False

	print 'Verifying class and image qty specs...'

	# Verify specified classes
	if class_list == ['<all>']:
		# Use all class subdirectories available
		class_list = os.listdir(imageset_dir)
		all_flag = True
	elif class_list == ['<flat>']:
		# No subdirectories
		class_list = []
	for a_dir in class_list:
		if a_dir not in os.listdir(imageset_dir):
			# Some specified directories are not in the imageset directory.
			missing_path = os.path.join(imageset_dir, a_dir)
			print 'error, %s does not exist.' % missing_path
			sys.exit(1)
	
	# Verify specified images
	if class_list == []:
		# no subdirectories, count images in imageset_dir
		imagecount = len(list_files(imageset_dir))
		if qty_list[0] == '<all>':
			qty_list[0] = imagecount
		elif int(qty_list[0]) > imagecount:
			# more images were specified than exists. assume maximum.
			print 'error, qty_list is greater than available images (%i>%i)'\
				% (int(qty_list[0]), imagecount)
			print 'assuming all images'
			qty_list[0] = imagecount
	else:
		if all_flag and qty_list[0] == '<all>':
			qty_list = ['all'] * len(class_list)
		for i in range(len(class_list)):
			imagecount = len(list_files(os.path.join(imageset_dir, \
				class_list[i])))
			if qty_list[i] == 'all':
				qty_list[i] = imagecount
				continue
			try:
				int(qty_list[i])
			except:
				# oops, entry is neither a number nor 'all'
				print '%s is not a valid parameter.' % qty_list[i]
				bad_config()
			if int(qty_list[i]) > imagecount:
				print 'error, qty_list is greater than available images \
					(class %s, %i>%i)' % (class_list[i], \
					int(qty_list[i]), imagecount)
				print 'assuming all images for class %s' % class_list[i]
				qty_list[i] = imagecount
	
	# At this point
	# class_list should be empty or a list of directories
	# qty_list should all be numbers.
	qty_list = [int(i) for i in qty_list]
	total_qty = sum(qty_list)

	# get list of imagepaths
	print 'Compiling imagepaths...'
	
	image_list = []
	label_list = []
	if class_list == []:
		image_list = list_files(imageset_dir)
		label_list = image_list
		image_list = [os.path.join(localdir, imageset_dir, image) for image in\
			image_list]
		image_list = image_list[:qty_list[0]]
		label_list = label_list[:qty_list[0]]
	else:
		for i in range(len(class_list)):
			cl_images = list_files(os.path.join(imageset_dir, class_list[i]))
			cl_labels = [class_list[i]+'-'+image for image in cl_images]
			cl_images = [os.path.join(localdir, imageset_dir, class_list[i],\
				image) for image in cl_images]
			cl_images = cl_images[:qty_list[i]]
			cl_labels = cl_labels[:qty_list[i]]
			if len(image_list) == 0:
				image_list = cl_images
				label_list = cl_labels
			else:
				image_list = np.concatenate((image_list, cl_images))
				label_list = np.concatenate((label_list, cl_labels))

	if len(label_list) != len(image_list):
		print 'fatal: image and label list are different lengths (%i!=%i).' \
			% (len(image_list), len(label_list))
		sys.exit(1)

	if total_qty != len(image_list):
		print 'Warning: spec image qty does not match actual qty (%i!=%i).' \
			% (total_qty, len(image_list))

	print 'Setting up network...'

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
	n_images = batch_size
	print 'Batch size is %i' % n_images
	n_channels = 3 #BGR
	net.blobs['data'].reshape(n_images, n_channels, 227, 227)
	
	# Set up batch filler
	# We need to make the input divisible by the batch size.
	# Since we're batching imagepaths, we'll just fill with the first image.
	fillerlen = len(image_list) % n_images
	batch_filler = [image_list[0]] * fillerlen
	image_list = np.concatenate((image_list, batch_filler))

	print 'Running network forward...'

	# feat_list[layer#][image#]
	feat_list = np.asarray([])
	bad_count = 0
	for b in range(len(image_list)/n_images):
		print "Processing batch %i of %i" % (b+1, len(image_list)/n_images)	
		
		# load batch
		curr_batch = image_list[b * n_images: b * n_images + n_images]

		for i in range(n_images):
			# try to load an image into caffe.
			try:
				# this will fail if the file is not an image.
				net.blobs['data'].data[i, ...] = transformer.preprocess( \
					'data', caffe.io.load_image(curr_batch[i]))
			except Exception as e:
				print e
				# Yikes, someone left a non-image file in the dataset.
				# This messes up our batching and dataoutput.
				# We'll try to handle this as gracefully as possible.
				bad_count += 1
				bad_index = b * n_images + i
				bad_label = label_list[bad_index]
				print "WARNING: Non-image file %s encountered." % bad_label
				print "Tagging label as BAD and loading synthetic data."
				label_list[bad_index] = bad_label + "BAD"
				net.blobs['data'].data[i, ...] = np.zeros((3, 227, 227))


		# Classify
		output = net.forward()

		# Extract features
		batch_list = []
		for k in range(len(layer_list)):
			a_layer = layer_list[k]
			try:
				# data should be shape (n_images, 4096)
				feat = net.blobs[a_layer].data.copy()
			except KeyError:
				print 'Warning: invalid layer %s, removed' % a_layer	
				layer_list.remove(a_layer)
				continue
			# We start building feat_list[layer#][image#][featval]
			if len(feat_list) == 0:
				batch_list.append(feat)
			else:
				feat_list[k] = np.concatenate((feat_list[k], feat), axis=0)

		if len(feat_list) == 0:
			feat_list = batch_list

	# feat_list[layer#][image#]

	# correct for batch buffer
	for k in range(len(layer_list)):
		feat_list[k] = feat_list[k][:total_qty]

	layer_count = len(feat_list)
	print '== Final Output =='
	print 'layers: %i' % layer_count
	print 'valid img: %i' % (total_qty - bad_count)
	print 'bad files: %i' % bad_count

		
	
	for i in range(len(feat_list)):
		outfilename = imageset_dir.split('/')[-1]+layer_list[i]+'.csv'

		print "exporting %s as %s" % (layer_list[i], 
						  os.path.join(out_dir,
							       outfilename))
		temp_array = np.vstack(feat_list[i])
		temp_list = []
		for j in range(temp_array.shape[0]):
			new_row = temp_array[j].tolist()
			new_row.insert(0, label_list[j])
			temp_list.insert(0, new_row)
		
		#export list as csv
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		with open(os.path.join(out_dir, outfilename),
			  'w') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
			wr.writerows(temp_list)

	return 1

if __name__ == "__main__":	

	print '==============='
	print '= Begin Debug ='
	print '==============='

	# check args
	if len(sys.argv) != 2:
		usage()
	elif len(sys.argv) == 2 and sys.argv[1] == '-h':
		usage()	
	configfile = sys.argv[1]
	
	# read in config file
	if not os.path.exists(configfile):
		print "Could not find config file", configfile
		sys.exit(1)

	# parse in config file
	print "Parsing config file..."
	config = ConfigParser.ConfigParser()
	config.read(configfile)
	
	imageset_dir = config.get('Files', 'imageset_dir')
	model_dir = config.get('Files', 'model_dir')
	out_dir = config.get('Files', 'out_dir')

	model_def = os.path.join(model_dir, config.get('Files', 'model_def'))
	if not os.path.exists(model_def):
		print 'Could not find model def file %s.' % model_def
		bad_config()

	model_weights = os.path.join(model_dir, config.get('Files', 'model_weights'))
	if not os.path.exists(model_weights):
		print 'Could not find model weights file %s.' % model_weights
		bad_config()
	
	mean_image = os.path.join(model_dir, config.get('Files', 'mean_image'))
	if not os.path.exists(mean_image):
		print 'Could not find mean image file %s.' % mean_image
		bad_config()

	batch_size = int(config.get('Params', 'batch_size'))

	GPU_enable = int(config.get('Params', 'GPU_enable'))
	if GPU_enable:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	layer_list = config.get('Params', 'layer_list')
	layer_list = layer_list.split(',')

	class_list = config.get('Params', 'class_list')
	class_list = class_list.split(',')
	
	qty_list = config.get('Params', 'qty_list')
	qty_list = qty_list.split(',')

	if len(class_list) != len(qty_list):
		print 'The class_list param list and qty_list param list must have the same length (%i != %i)' % (len(class_list), len(qty_list))
		bad_config()
	
	export_csv(model_def, model_weights, mean_image, 
		layer_list, class_list, qty_list, imageset_dir, batch_size)




