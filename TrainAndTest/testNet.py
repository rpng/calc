import caffe
import math
import numpy as np
from matplotlib import pyplot as plt, rcParams
from matplotlib.font_manager import FontProperties
import cv2
from sklearn.metrics import precision_recall_curve, auc
from time import time
import re
from os import path, getcwd, listdir, makedirs
import sys

first_it = True
A = None

def smooth_pr(prec, rec):
	"""
	Smooths precision recall curve according to TREC standards. Evaluates max precision at each 0.1 recall. Makes the curves look nice and not noisy
	"""

	n = len(prec)
	m = 11
	p_smooth = np.zeros((m), dtype=np.float)
	r_smooth = np.linspace(0.0, 1.0, m) 
	for i in range(m):
		j = np.argmin( np.absolute(r_smooth[i] - rec) ) + 1
		p_smooth[i] = np.max( prec[:j] )
			
	return p_smooth, r_smooth	

def check_match(im_lab_k, db_lab, num_include):
	"""
	Check if im_lab_k and db_lab are a match, i.e. the two images are less than or equal to
	num_include frames apart. The correct num_include to use depends on the speed of the camera, both for frame rate as well as physical moving speed.
	"""	
	if num_include == 1:
		if db_lab ==im_lab_k:
			return True
	else:				
		# This assumes that db_lab is a string of numerical characters, which it should be	
		#print int(db_lab)-num_include/2, "<=", int(im_lab_k), "<=", int(db_lab)+num_include/2, "?"
		if (int(db_lab)-num_include/2) <= int(im_lab_k) and int(im_lab_k) <= (int(db_lab)+num_include/2):
			return True

	return False

def computeForwardPasses(nets, alexnet, im, transformer, transformer_alex, resize_net):
	"""
	Compute the forward passes for CALC and optionallly alexnet
	"""

	img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	alex_conv3 = None
	t_alex = -1

	imcp = np.copy(im) # for AlexNet
	
	if im.shape[2] > 1:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	if not resize_net:
		im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC)
	else:
		transformer = caffe.io.Transformer({'X1':(1,1,im.shape[0],im.shape[1])})	
		transformer.set_raw_scale('X1',1./255)
		for net in nets:
			x1 = net.blobs['X1']
			x1.reshape(1,1,im.shape[0],im.shape[1])
			net.reshape()
	descr = []
	t_calc = []
	for net in nets:
		t0 = time()
		net.blobs['X1'].data[...] = transformer.preprocess('X1', im)
		net.forward()
		d = np.copy(net.blobs['descriptor'].data[...])
		t_calc.append(time() - t0)
		d /= np.linalg.norm(d)
		descr.append(d)

	if alexnet is not None:
		im2 = cv2.resize(imcp, (227,227), interpolation=cv2.INTER_CUBIC)
		t0 =  time()
		alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', im2)
		alexnet.forward()
		alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
		alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
		global first_it
		global A
		if first_it:
			np.random.seed(0)
			A = np.random.randn(descr[0].size, alex_conv3.size) # For Gaussian random projection
			first_it = False
		alex_conv3 = np.matmul(A, alex_conv3)
		alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
		t_alex = time() - t0
		alex_conv3 /= np.linalg.norm(alex_conv3)

	return descr, alex_conv3, t_calc, t_alex



def get_prec_recall(net_def_path='proto/calc_deploy.prototxt', net_model_path='model/calc_iter_100000.caffemodel', data_path="test_data/CampusLoopDataset", num_include=7, title='Precision-Recall Curve', resize_net=False, alexnet_proto_path=None, alexnet_weights=None):
	
	"""
	Input: 
	 
	net_def_path='proto/calc_deploy.prototxt': Caffe network definition file

	net_model_path='model/calc_iter_100000.caffemodel': .caffemodel weights file

	data_path="test_data/CampusLoopDataset",: Path to data with corresponding images (with corresponding file names) in a <data-dir>/live and <data-dir>/memory
	
	resize_net - if True, the net will resize fot he native image size, otherwise, the images
		will be resized to the native net input size of 120x160
	"""

	caffe.set_mode_gpu()
	caffe.set_device(0)
	
	dbow = None
	# Check is DBoW2 fork is installed 
	if path.isfile("ThirdParty/DBoW2/build/pydbow2.so"):
		sys.path.append("ThirdParty/DBoW2/build")
		from pydbow2 import PyDBoW2
		dbow = PyDBoW2("ThirdParty/DBoW2/Vocabulary/ORBvoc.txt")	
		
	nets = []
	for m_path in net_model_path:
		nets.append(caffe.Net(net_def_path,1,weights=m_path))

	database = [] # stored pic descriptors
	database_labels = [] # the image labels	

	alexnet = None
	transformer_alex = None
	if alexnet_proto_path is not None:
		db_alex = []
		alexnet = caffe.Net(alexnet_proto_path,1,weights=alexnet_weights)

		transformer_alex = caffe.io.Transformer({'data':(1,3,227,227)})	
		transformer_alex.set_raw_scale('data',1./255)
		transformer_alex.set_transpose('data', (2,0,1))
		transformer_alex.set_channel_swap('data', (2,1,0))

	mem_path = data_path + "/memory"
	live_path = data_path + "/live"

	print "memory path: ", mem_path
	print "live path: ", live_path

	mem_files = [path.join(mem_path, f) for f in listdir(mem_path)]
	live_files = [path.join(live_path, f) for f in listdir(live_path)]

	# Use caffe's transformer
	transformer = caffe.io.Transformer({'X1':(1,1,120,160)})	
	transformer.set_raw_scale('X1',1./255)
	
	# same HOG params used to train calc
	hog = cv2.HOGDescriptor((16, 32), (16,16), (16,16), (8,8), 2,1)
	db_hog = []

	t_calc = []
	t_alex = []

	for fl in mem_files:	
		im = cv2.imread(fl)
		print "loading image ", fl, " to database"
		descr, alex_conv3, t_r, t_a = computeForwardPasses(nets, alexnet, im, transformer, transformer_alex, resize_net)		
		t_calc.append(t_r)
		t_alex.append(t_a)
		if alexnet is not None:
			db_alex.append(alex_conv3)		
		database.append(descr) 		
	
		# Use the image sequence number as the label.This assumes that live and memory have the image numbers synce with location, like the Garden Point dataset, or an stereo dataset	
		database_labels.append(re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1))

		d_hog = hog.compute(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY), (160, 120), interpolation = cv2.INTER_CUBIC))
		db_hog.append(d_hog/np.linalg.norm(d_hog))
		
		if dbow is not None:
			dbow.addToDB(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) 


	#print database

	correct = np.zeros((len(nets),len(live_files)),dtype=np.uint8) # the array of true labels of loop closure for precision-recall curve for each net
	
	scores = np.zeros((len(nets),len(live_files)))  # Our "probability function" that  simply uses 1-l2_norm

	correct_alex = []
	scores_alex = []

	correct_hog = []
	scores_hog = []

	correct_dbow = []
	scores_dbow = []

	k=0
	for fl in live_files:	
		im_label_k = re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1)
		im = cv2.imread(fl)

		descr, alex_conv3, t_r, t_a = computeForwardPasses(nets, alexnet, im, transformer, transformer_alex, resize_net)		
		t_calc.append(t_r)
		t_alex.append(t_a)
			
		d_hog = hog.compute(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160, 120), interpolation = cv2.INTER_CUBIC))
		d_hog /= np.linalg.norm(d_hog)

		max_sim = -1.0 * np.ones(len(nets))
		max_sim_alex = -1.0
		max_sim_hog = -1.0	

		i_max_sim = -1 * np.ones(len(nets), dtype=np.int32)
		i_max_sim_alex = -1
		i_max_sim_hog = -1

		for i in range(len(database)):
			for j in range(len(nets)):
				curr_sim = np.dot(descr[j], database[i][j].T) # Normalizd vectors means that this give cosine similarity
				if curr_sim > max_sim[j]: 
					max_sim[j] = curr_sim
					i_max_sim[j] = i

			if alexnet is not None:
				curr_sim_alex = np.squeeze(np.dot(alex_conv3, db_alex[i].T)) 
				if curr_sim_alex > max_sim_alex: 
					max_sim_alex = curr_sim_alex
					i_max_sim_alex = i

			curr_sim_hog = np.squeeze(np.dot(d_hog.T,  db_hog[i])) 
			if curr_sim_hog > max_sim_hog: 
				max_sim_hog = curr_sim_hog
				i_max_sim_hog = i

			if dbow is not None:
				dbow

		scores[:,k] = max_sim 
	
		for j in range(len(nets)):	
			db_lab = database_labels[i_max_sim[j]]  
			if check_match(im_label_k, db_lab, num_include):
				correct[j][k] = 1
			# else already 0
			print "Proposed match calc:", im_label_k, ", ", database_labels[i_max_sim[j]], ", score = ", max_sim[j], ", Correct =", correct[j][k]

		if alexnet is not None:		
			scores_alex.append( max_sim_alex )	
			db_lab_alex = database_labels[i_max_sim_alex]  
			if check_match(im_label_k, db_lab_alex, num_include):
				correct_alex.append(1)
			else:
				correct_alex.append(0)			

			print "Proposed match AlexNet:", im_label_k, ", ", database_labels[i_max_sim_alex], ", score = ", max_sim_alex,", Correct =", correct_alex[-1]

		scores_hog.append( max_sim_hog )	
		db_lab_hog = database_labels[i_max_sim_hog]  
		if check_match(im_label_k, db_lab_hog, num_include):
			correct_hog.append(1)
		else:
			correct_hog.append(0)			
		print "Proposed match HOG:", im_label_k, ", ", database_labels[i_max_sim_hog], ", score = ", max_sim_hog,", Correct =", correct_hog[-1]

	
		if dbow is not None:
			i_max_score_dbow, max_score_dbow = dbow.getClosestMatch(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
			scores_dbow.append(max_score_dbow)			
			if check_match(im_label_k, database_labels[i_max_score_dbow], num_include):
				correct_dbow.append(1)
			else:
				correct_dbow.append(0)			

			print "Proposed match DBoW2:", im_label_k, ", ", database_labels[i_max_score_dbow], ", score = ", max_score_dbow,", Correct =", correct_dbow[-1]
	
		print "\n"
		k += 1

	precisions = []
	recalls = []
	threshold = -1.0

	for j in range(len(nets)):
		precision, recall, threshold = precision_recall_curve(correct[j], scores[j])
		precisions.append(precision)
		recalls.append(recall)
		if len(nets) == 1: # Only get threshold if there's one net. Otherwise we're just coparing them and don't care about a threshold yet
			perf_prec = abs(precision[:-1] - 1.0) <= 1e-6
			if np.any(perf_prec):	
				# We want the highest recall rate with perfect precision as our a-priori threshold
				threshold = np.min(threshold[perf_prec]) # get the largest threshold so that presicion is 1
	print "\nThreshold for max recall with 1.0 precision = %f" % (threshold) 
	precision_alex = None
	recall_alex = None
	if alexnet is not None:
		precision_alex, recall_alex, thresholds_alex = precision_recall_curve(correct_alex, scores_alex)

	precision_dbow = None
	recall_dbow = None
	if dbow is not None:
		precision_dbow, recall_dbow, thresholds_dbow = precision_recall_curve(correct_dbow, scores_dbow)

	precision_hog, recall_hog, thresholds_hog = precision_recall_curve(correct_hog, scores_hog)

	print "Mean calc compute time = ", np.matrix.sum(np.matrix(t_calc))/ np.size(np.array(t_calc))

	return precisions, recalls, precision_alex, recall_alex, precision_hog, recall_hog, precision_dbow, recall_dbow


def plot(net_def_path='proto/deploy.prototxt', net_model_path=['model/calc.caffemodel'], data_path="test_data/CampusLoopDataset", num_include=7, title='Precision-Recall Curve', resize_net=False, alexnet_proto_path=None, alexnet_weights=None):
	"""
	Plot the precision recall curve to compare CALC to other methods, or cross validate different iterations of a CALC model.
	"""
		
	t0 = time()

	if isinstance(net_model_path, basestring): # someone only inputed a single string, so make it a list so that this code works
		net_model_path = [net_model_path]


	precisions, recalls, precision_alex, recall_alex, precision_hog, recall_hog, precision_dbow, recall_dbow = get_prec_recall(net_def_path, net_model_path, data_path, num_include, title, resize_net, alexnet_proto_path, alexnet_weights)

	rcParams['font.sans-serif'] = 'DejaVu Sans'
	rcParams['font.weight'] = 'bold'
	rcParams['axes.titleweight'] = 'bold'	
	rcParams['axes.labelweight'] = 'bold'	
	rcParams['axes.labelsize'] = 'large'	
	rcParams['figure.figsize'] = [8.0, 4.0]	
	rcParams['figure.subplot.bottom'] = 0.2	
	plots = path.join(getcwd(), "plots")
	rcParams['savefig.directory'] = plots
	if not path.isdir(plots):
		makedirs(plots)

	lines = ['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
	
	ax = plt.gca()
	best_auc = -1 
	lab_best_auc = ""
	handles = []
	for j in range(len(precisions)):
		p_smooth, r_smooth = smooth_pr(precisions[j], recalls[j])
		curr_auc = auc(r_smooth, p_smooth)
		label = re.split('.caffemodel',re.split('/',net_model_path[j])[-1])[0]
		if len(precisions) < 4:
			label += ' (AUC=%0.2f)' % (curr_auc)
		
		calc_plt, = ax.plot(r_smooth, p_smooth, '-', label=label, linewidth=2)

		handles.append(calc_plt)
		if j==0 or curr_auc > best_auc:
			lab_best_auc = label
			best_auc = curr_auc	
	# Only tell the user the most accurate net if they loaded more than one!
	if len(precisions) > 1:
		print("Model with highest AUC:", lab_best_auc)


	print "\n\n\n\n"
	
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0, 1])
	plt.xlim([0, 1])
	plt.title(title)

	if precision_alex is not None:
		p_smooth, r_smooth = smooth_pr(precision_alex, recall_alex)
		lab = 'AlexNet (AUC=%0.2f)' % (auc(r_smooth, p_smooth))
		alex_plt, = ax.plot(r_smooth, p_smooth, '--', label=lab, linewidth=2)
		handles.append(alex_plt)
	if precision_dbow is not None:
		p_smooth, r_smooth = smooth_pr(precision_dbow, recall_dbow)
		dbow_plt, = ax.plot(r_smooth, p_smooth, '-.', label='DBoW2 (AUC=%0.2f)' % (auc(r_smooth, p_smooth)), linewidth=2)
		handles.append(dbow_plt)
	p_smooth, r_smooth = smooth_pr(precision_hog, recall_hog)
	lab = 'HOG (AUC=%0.2f)' % (auc(r_smooth, p_smooth))
	hog_plt, = ax.plot(r_smooth, p_smooth, '-.', label=lab, linewidth=2)
	handles.append(hog_plt)
	fontP = FontProperties()
	fontP.set_size('small')
	leg = ax.legend(handles=handles, fancybox=True, ncol = (1 + int(len(precisions)/30)), loc='best', prop=fontP)
	leg.get_frame().set_alpha(0.5) # transluscent legend :D
	leg.draggable()
	for line in leg.get_lines():
		line.set_linewidth(3)
	
	print "Elapsed time = ", time()-t0, " sec"
	plt.show()	

def plot_var_db_size(calc_fl_name, dbow_fl_name):
	'''
	For use with files output from Thirdparty/DBoW2/build/vary-db-size and ../DeepLCD/build/vary-db-size.
	Files must be created using the same dataset, so that they are the same length
	'''

	rcParams['font.sans-serif'] = 'DejaVu Sans'
	rcParams['font.weight'] = 'bold'
	rcParams['axes.titleweight'] = 'bold'	
	rcParams['axes.labelweight'] = 'bold'	
	rcParams['axes.labelsize'] = 'large'	
	rcParams['figure.figsize'] = [8.0, 4.0]	
	rcParams['figure.subplot.bottom'] = 0.2	
	plots = path.join(getcwd(), "plots")
	rcParams['savefig.directory'] = plots
	if not path.isdir(plots):
		makedirs(plots)
	ax = plt.gca()
	handles = []
	calc_fl = open(calc_fl_name, "r")	
	calc_str = calc_fl.read().split('\n')
	m_calc = np.array(re.split(' ', calc_str[0])[:-1]).astype(np.float) # db sizes
	t_calc = np.array(re.split(' ', calc_str[1])[:-1]).astype(np.float) # mean query times
	calc_plt, = ax.plot(m_calc, t_calc, label='Ours', linewidth=2)
	handles.append(calc_plt)	

	dbow_plt = None
	if dbow_fl_name is not None:
		dbow_fl = open(dbow_fl_name, "r")	
		dbow_str = dbow_fl.read().split('\n')
		m_dbow = np.array(re.split(' ', dbow_str[0])[:-1]).astype(np.float) # db sizes
		t_dbow = np.array(re.split(' ', dbow_str[1])[:-1]).astype(np.float) # mean query times
		dbow_plt, = ax.plot(m_dbow, t_dbow, '--', label='DBoW2', linewidth=2)
		handles.append(dbow_plt)
	plt.xlabel('Database Size')
	plt.ylabel('Mean Query Time (ms)')
	#plt.ylim([0, ])
	plt.xlim([0, np.max(m_calc)])
	plt.title("Querying with Varying Database Size")

	fontP = FontProperties()
	fontP.set_size('small')
	leg = ax.legend(handles=handles, loc='upper left', prop=fontP,fancybox=True)
	leg.get_frame().set_alpha(0.5) # transluscent legend :D
	leg.draggable()
	for line in leg.get_lines():
		line.set_linewidth(3)
	plt.show()	


def view_forward_pass(im1_fl, im2_fl, net_def_path='proto/deploy.prototxt', net_model_path='model/Ours.caffemodel'):

	"""
	View the forward pass of an image through the deployed net. 
	"""
	from matplotlib import rcParams
	caffe.set_mode_cpu()
	net = caffe.Net(net_def_path,1,weights= net_model_path)
	im1 = cv2.resize(cv2.cvtColor(cv2.imread(im1_fl), cv2.COLOR_BGR2GRAY), (160,120))
	im2 = cv2.resize(cv2.cvtColor(cv2.imread(im2_fl), cv2.COLOR_BGR2GRAY), (160,120))
	# Use caffe's transformer
	transformer = caffe.io.Transformer({'X1':(1,1,120,160)})	
	transformer.set_raw_scale('X1',1./255)
	net.blobs['X1'].data[...] = transformer.preprocess('X1', im1)
	net.forward()
	relu13 = np.copy(net.blobs['relu3'].data[0,0,:,:])

	net.blobs['X1'].data[...] = transformer.preprocess('X1', im2)
	net.forward()
	relu23 = net.blobs['relu3'].data[0,0,:,:]
		
	plt.axis('off')
	plt.imshow(relu13)
	plt.show()	

	plt.axis('off')
	plt.imshow(relu23)
	plt.show()













