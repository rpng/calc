from glob import glob
import cv2
import numpy as np
import caffe
import os
from random import Random
import lmdb
import math
import unittest
from psutil import virtual_memory
import time
from sys import platform
import click # progress bar
import re
from makeNet import create_net, train, moveModel
from matplotlib import pyplot as plt

class bcolors:
	"""
	The colors are just for fun
	"""

	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[0;92m'
	ENDC = '\033[0m'
	PURPLE = '\033[0;95m'	

def writeFileList(dirNameArr):
	"""
	Returns the python list object of the files under a directory name for processing later
	"""
	print bcolors.OKGREEN + '\nParsing files ...\n' 

	if isinstance(dirNameArr, basestring): # someone only inputed a single string, so make it a list so that this code works

		dirNameArr = [dirNameArr]


	files_list = [] # list of all files with full path
	for dirName in dirNameArr: 
	# loop through all files in the list of directory names inputted. This is useful for multiple datasets	
		with click.progressbar(os.walk(dirName), label="Parsing files in "+dirName) as bar:
			for dirname, dirnames, filenames in bar:
				for filename in filenames:
					if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.bmp') or filename.endswith('.tiff'):	
						fileName = glob(os.path.join(dirname, filename)) 
						files_list += fileName

	print bcolors.ENDC 
		
	return files_list
	
def randPerspectiveWarp(im, w, h, r, ret_pts=False):

	"""
	Applies a pseudo-random perspective warp to an image. 

	input: 

	im - the original image

	h - image height
	
	w - image width

	r - Random instance

	returns:

	im_warp - the warped image

	ret_pts - if True, return the points generated 
	""" 

	# Generate two pseudo random planes within tolerances for the projective transformation of the original image
	# Each point is from the center half of its respective x y quandrant. openCV getPerpectiveTransform expects [Q2, Q3, Q1, Q4] for points in each image quandrant, so that it 
	# the iteration order here. Note that 0,0 is the top left corner of the picture. Additionally, we can only perform a tranformation to zoom in, since exprapolated pixels
	# look unnatural, and will ruin the similarity between the two images.

	# limits for random number generation
	minsx = [ 0, 3*w/4 ]
	maxsx = [ w/4, w ]
	minsy= [ 0, 3*h/4 ]
	maxsy = [ h/4, h ]


	pts_orig = np.zeros((4, 2), dtype=np.float32) # four original points
	pts_warp = np.zeros((4, 2), dtype=np.float32) # points for the affine transformation. 

	# fixed point for the first plane	
	pts_orig[0, 0] = 0
	pts_orig[0, 1] = 0
	
	pts_orig[1, 0] = 0
	pts_orig[1, 1] = h

	pts_orig[2, 0] = w
	pts_orig[2, 1] = 0

	pts_orig[3, 0] = w
	pts_orig[3, 1] = h

	# random second plane
	pts_warp[0, 0] = r.uniform(minsx[0], maxsx[0])
	pts_warp[0, 1] = r.uniform(minsy[0], maxsy[0])
	
	pts_warp[1, 0] = r.uniform(minsx[0], maxsx[0])
	pts_warp[1, 1] = r.uniform(minsy[1], maxsy[1])

	pts_warp[2, 0] = r.uniform(minsx[1], maxsx[1])
	pts_warp[2, 1] = r.uniform(minsy[0], maxsy[0])

	pts_warp[3, 0] = r.uniform(minsx[1], maxsx[1])
	pts_warp[3, 1] = r.uniform(minsy[1], maxsy[1])

	# compute the 3x3 transform matrix based on the two planes of interest
	T = cv2.getPerspectiveTransform(pts_warp, pts_orig)

	# apply the perspective transormation to the image, causing an automated change in viewpoint for the net's dual input
	im_warp = cv2.warpPerspective(im, T, (w, h))
	if not ret_pts:
		return im_warp
	else: 
		return im_warp, pts_warp

def showImWarpEx(im_fl, save):
	"""
	Show an example of warped images and their corresponding four corner points.
	"""

	im = cv2.resize(cv2.cvtColor(cv2.imread(im_fl), cv2.COLOR_BGR2GRAY), (256,int(120./160*256)))
	r = Random(0)
	r.seed(time.time())
	h, w = im.shape
	im_warp, pts_warp = randPerspectiveWarp(im, w, h, r, ret_pts=True) # get the perspective warped picture	
	
	pts_orig = np.zeros((4, 2), dtype=np.float32) # four original points
	ofst = 3
	pts_orig[0, 0] = ofst
	pts_orig[0, 1] = ofst	
	pts_orig[1, 0] = ofst
	pts_orig[1, 1] = h-ofst
	pts_orig[2, 0] = w-ofst
	pts_orig[2, 1] = ofst
	pts_orig[3, 0] = w-ofst
	pts_orig[3, 1] = h-ofst

	kpts_warp = []
	kpts_orig = []
	matches = []

	pts_rect = np.zeros((4, 2), dtype=np.float32) # for creating rectangles
	pts_rect[0, 0] = w/4
	pts_rect[0, 1] = h/4	
	pts_rect[1, 0] = w/4
	pts_rect[1, 1] = 3*h/4
	pts_rect[2, 0] = 3*w/4
	pts_rect[2, 1] = h/4
	pts_rect[3, 0] = 3*w/4
	pts_rect[3, 1] = 3*h/4
	if save: # save orig before placing rectangles on it
		cv2.imwrite("Original.jpg", im)

	for i in range(4):
		kpts_warp.append(cv2.KeyPoint(pts_warp[i,0], pts_warp[i,1], 0))
		kpts_orig.append(cv2.KeyPoint(pts_orig[i,0], pts_orig[i,1], 0))
		matches.append(cv2.DMatch(i,i,0))
		im = cv2.rectangle(im, (pts_orig[i,0], pts_orig[i,1]), (pts_rect[i,0], pts_rect[i,1]), (255,255,255), thickness=2)	
	draw_params = dict(matchColor=(0,0,250),flags = 4)
	out_im = cv2.drawMatches(im, kpts_warp, im_warp, kpts_orig, matches, None, **draw_params)
	plots = os.path.join(os.getcwd(), "plots")
	from matplotlib import rcParams
	rcParams['savefig.directory'] = plots
	if not os.path.isdir(plots):
		os.makedirs(plots)
	plt.imshow(out_im)
	plt.axis('off')
	plt.show()
	if save:
		cv2.imwrite("Warped.jpg", im_warp)
		print "Images saved in current directory"
def calcNumBuff(w, h, n, n_comp, mem):
	""" 
	calculate the minimum number of buffers to use based on the capacity of system ram and gpu memory
	"""

	# calculate the number of buffers to use based on the data size and the available RAM
	# Additionally, only use about 7/8 of RAM for buffers dues to other overheads (mainly from PCA), and add 1 extra buffer in case the calculated number is zero

	# RAM in bytes
	ram_mem = mem.available

	bytePerFloat = 4
	bytePerInt = 1

	# images are uint8, descriptors are floats
	bytesInRAMFromData = int( n * (n_comp * bytePerFloat + w*h*bytePerInt) )

	min_buff_ram = 1 + int( 3.0 * bytesInRAMFromData / ram_mem / 4.0) # min number of buffers to take up a little less than 75% RAM per buffer, plus one in case it's zero

	# return the max number of buffers out of the two so that we dont overload either RAM or GPU memory. This makes this code portable to other machines, but the results will vary
	return min_buff_ram, bytesInRAMFromData

def decideWaitForMem(initBytesNeeded, percentDone, mem):
	'''
	Return true if need to wait for more memory. Return false otherwise
	'''
	needWaitForMem = False	
	if (1.0 - percentDone) * initBytesNeeded > mem.available: # note that percent done is in [0,1]
		needWaitForMem = True

	return needWaitForMem


def writeDatabase(outDBNames, files_list, w, h, data_root="", gpu_id=0, prev_model_basename="", test_db=False, debugFlag=False, trainAfter=False):

	"""
	Creates two tensors of image matrices. The images are read from the filenames in files_list, resized, converted to grayscale if they are color.
	The images are shuffled to avoid statistical issues with caffe, then randomly swapped.
	X1 contains just images from each pair, and X2 contains HOG descriptors from the other image of each pair 
	The end product is created in buffers and written to two LMDBs for X1 and X2 accordingly
	"""

	if (not data_root=="") and (not data_root.endswith('/')): # expect directory name to have '/' appended
		data_root += '/'
	
	if not 'linux' in platform:
		raise Exception('Error: only UNIX machines are currently supported')

	print '\npreparing to transform images and write databases ...\n' 

	n = len(files_list) # number of samples

	n_comp = 3648 # HOG vector length

	mem = virtual_memory()
	num_buff, bytesNeeded = calcNumBuff(w, h, n, n_comp, mem) # number of buffers for database writing
	if debugFlag:
		num_buff = 3
		bytesNeeded = 0
                	
	n_per_buff = int(math.ceil(n/num_buff)) 
	print bcolors.PURPLE + "Number of buffers: ", num_buff, ", Images per buffer: ", n_per_buff, ", Total image count: " + str(n) + '\n\n\n\n\n\n\n' + bcolors.ENDC

	r = Random(0) # make a random number generator instance seeded with 0	
	if test_db:
		plt.ion()	
	inds = range(n)
	# shuffling indices will slow down the array accessing process,
	# but will take away any relationship between adjacent images, making the model better
	r.shuffle(inds) # note that shuffle works in place
	if not os.path.isdir(data_root+"train_data"):
		os.makedirs(data_root+"train_data")
	# prepare the max database size. There is no harm in making it too big, since this is only the cap, not the actual size. If disk space runs out, it will throw an error and crash anyways
	map_size = 1024**4 # 1 TB
	chan = 1
	first_buff_flag = True
	im_count_tot = 0 # total number of pictures
	i_to_show = r.randint(0,n_per_buff-1)
	hog = cv2.HOGDescriptor((16, 32), (16,16), (16,16), (8,8), 2,1)	
	X1_db_name = data_root + "train_data/" + outDBNames[0]
	X2_db_name = data_root + "train_data/" + outDBNames[1]

	with click.progressbar(range(num_buff), label="Total Progress") as bigBar:
		for j in bigBar:
			##### Database writing #################################
			db1 = lmdb.open(X1_db_name, map_size=map_size)
			db2 = lmdb.open(X2_db_name, map_size=map_size)
			k = 0 # index in X row
			txn1 = db1.begin(write=True, buffers=True) # start a new transaction for the database
			txn2 = db2.begin(write=True, buffers=True)
			with click.progressbar(inds[(j*n_per_buff):((j+1)*n_per_buff)], label=("Progress in buffer "+str(j+1)+" out of " + str(num_buff))) as bar:
				for i in bar: # index in files_list, which is n long
					im_file = files_list[i]
					im = cv2.imread(im_file)
					while im is None: # Some images get corrupted sometimes. Check for this so that it doesnt crash a multi-day running process (sigh)
						print "\n\n\nSkipping corrupted image:",im_file, ". Bootstrapping random image from dataset\n\n\n"
						im_file = files_list[r.randint(0, n-1)]
						im = cv2.imread(im_file)
	
					if len(im.shape) > 2: # if color image, convert to grayscale
						im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

					im = cv2.resize(im, (w, h), interpolation = cv2.INTER_CUBIC)
					im_warp = randPerspectiveWarp(im, w, h, r) # get the perspective warped picture	
					r.seed(i) # adds extra randomness, but is still reproduceable with the same dataset
					# image processing function. Needs random number generator
					#im, im_warp = preprocess(im, im_warp, r)				
					if test_db and i==i_to_show: # only show the last image. If we show all of them, computation is VERY slow
						plt.subplot(121),plt.imshow(im, cmap='gray'),plt.title('Original, Brightness')
						plt.subplot(122),plt.imshow(im_warp, cmap='gray'),plt.title('Perspective Changed')
						plt.pause(10) # only show it for 10 seconds incase the user walks away.
						plt.close()			
								
					# randomly choose whether the original or the transformed image go in X1/X2 respectively. This prevents bias in the model since we can only zoom in
					switchFlag = r.randint(0,1)
					if switchFlag:
						im1 = im_warp
						des = hog.compute(cv2.resize(im, (160,120)))
					else:
						im1 = im
						des = hog.compute(cv2.resize(im_warp, (160,120)))

					str_id = '{:08}'.format(im_count_tot).encode('ascii') # we only need one str_id as well, since they should be the same for corresponding images

					datum1 = caffe.proto.caffe_pb2.Datum()
					datum1.channels = chan
					datum1.width = w
					datum1.height = h
					datum1.data = im1.tobytes() 
				
					txn1.put(str_id, datum1.SerializeToString()) # add it to database1
					datum2 = caffe.proto.caffe_pb2.Datum()
					datum2.channels = 1
					datum2.width = 1
					datum2.height = n_comp
					datum2.data = np.reshape(des, (n_comp)).tobytes() 
						
					txn2.put(str_id, datum2.SerializeToString()) # add it to database2
					k += 1
					im_count_tot += 1
					if im_count_tot == n-1:
						break
				# end for i in inds[(j*n_per_buff):((j+1)*n_per_buff)]:
			txn1.commit()
			txn2.commit()
		# end for j in range(num_buff)

	if trainAfter:
		if not debugFlag:
			batch_sz = 256 # Use 256 batch size for easy testing on small GPU, can increase to train faster. For multi gpu use the bash script with the caffe executable
			n_epochs = 43 # We gave our net about 42.25 epochs, so just ceil that number to get 43 epochs
			its = int(n_epochs * n / batch_sz) # calculate number of its based on data size!! 
		else:		
			its = 12
		
		snapshot_prefix = '"calc"' # prefix for current model

		print "\nDefining net with new database\n"

		# define the current train net. Note that the [11:] drops the 'train_data/' prefix
		create_net(X1_db_name[11:], X2_db_name[11:], max_iter=str(its), data_root=data_root, debugFlag=debugFlag)

		print "\nInitializing Training"

		# train the net with the current data
		train("proto/solver.prototxt", GPU_ID=gpu_id)

		moveModel(model_dir="calc_" +  time.strftime("%d-%m-%Y_%I%M%S")) # move all the model files to a directory 
		


def launch(w, h, dirNameArr, outDBNames, data_root="", gpu_id=0, test_db=False, debugFlag=False, trainAfter=False, batch_size=256):
	"""
	Launch the database writing with the widht and height desired in the database,
	the list of directory names for the dataset(s), and the names of the output directories
	for X1 and X2, respectively 
	"""
	t0 = time.time()

	print "\n\n\n\n\n\n" + bcolors.OKBLUE

	# This is just for fun	
	print """   ******      **     **         ****** 
  **////**    ****   /**        **////**
 **    //    **//**  /**       **    // 
/**         **  //** /**      /**       
/**        **********/**      /**       
//**    **/**//////**/**      //**    **
 //****** /**     /**/******** //****** 
  //////  //      // ////////   //////"""
	print bcolors.ENDC + "\n\n\n\n\n"

	files_list = writeFileList(dirNameArr) # recursively  glob the file names.

	writeDatabase(outDBNames, files_list, w, h, data_root=data_root, gpu_id=gpu_id, test_db=test_db, debugFlag=debugFlag, trainAfter=trainAfter) # write the two databases

	print '\n\ndone'	
	t1 = time.time()
	print '\n\nDatabase writing + optional training time: ', (t1-t0) / 60 / 60, ' hours'
