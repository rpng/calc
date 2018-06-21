import caffe
import os
from shutil import copy, move
from glob import glob
import re
from time import time

def conv(inpt, ks, stride=1, num_out=1, group=1, pad=0):
	"""
	Convolution layer definition. One filter is always the output's third dimmension, and there is never any zero pad, since we want a compact 
	descriptor from the encoded image.
	"""
	conv = caffe.layers.Convolution(inpt, convolution_param=dict(kernel_size=ks, stride=stride, num_output=num_out, pad=pad, group=group), weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='gaussian',std=0.01)) 
	return conv

def max_pool(inpt, ks, stride=1):
	"""
	Max pooling layer definition
	"""
	pool = caffe.layers.Pooling(inpt, pool=caffe.params.Pooling.MAX, kernel_size=ks, stride=stride)
	return pool

def fc(inpt, nout):
	"""
	Fully conneted layer with variable output size for decoding layers
	"""
	fc = caffe.layers.InnerProduct(inpt, num_output=nout, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='gaussian', std=0.01))
	return fc

def sig(inpt):
	"""
	Sigmoid activation layer
	"""
	sig = caffe.layers.Sigmoid(inpt)
	return sig

def relu(inpt):
	relu = caffe.layers.ReLU(inpt)
	return relu

def calc(X1_lmdb, X2_lmdb, data_root="", batch_size=256):
	"""
	Network definition and writing to prototxt from lmdb image database. Level Databases are very fast for training use.
	"""
	print 'Defining calc ...'
	
	n = caffe.NetSpec()
	
	X1_path = "train_data/" + X1_lmdb
	X2_path = "train_data/" + X2_lmdb
	
	if not (data_root=="/" or data_root==""):
		X1_path = data_root + X1_path
		X2_path = data_root + X2_path

	n.X1 = caffe.layers.Data(source=X1_path, backend=caffe.params.Data.LMDB, batch_size=batch_size, ntop=1, transform_param=dict(scale=1./255)) 	
		
	n.X2 = caffe.layers.Data(source=X2_path, backend=caffe.params.Data.LMDB, batch_size=batch_size, ntop=1, transform_param=dict(scale=1./255))  	

	# encode
	n.conv1 = conv(n.X1, 5, stride=2, num_out=64, pad=4)
	n.relu1 = relu(n.conv1)
	n.pool1 = max_pool(n.relu1, 3, stride=2)
	n.norm1 = caffe.layers.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75) 	

	n.conv2 = conv(n.norm1, 4, num_out=128, pad=2)
	n.relu2 = relu(n.conv2)
	n.pool2 = max_pool(n.relu2, 3, stride=2)
	n.norm2 = caffe.layers.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75) 	

	n.conv3 = conv(n.norm2, 3, num_out=4) # 4x14x19 , p = 1064 descriptor
	n.relu3 = relu(n.conv3)

	n.descriptor = caffe.layers.Flatten(n.relu3) # output of the deployed net after we throw out the other half

	#decode
	n.fc4 = fc(n.descriptor, 1064)
	n.sig4 = sig(n.fc4)

	n.fc5 = fc(n.sig4, 2048)
	n.sig5 = sig(n.fc5)

	n.fc6 = fc(n.sig5, 3648)
	n.X2_comp_hat = sig(n.fc6) # decoded reduced image vector. Use sig to normalize to [0,1] for better results

	################ Loss #######################################

	n.loss = caffe.layers.EuclideanLoss(n.X2_comp_hat, caffe.layers.Flatten(n.X2)) #special_l2_loss(n.X2_comp_hat, caffe.layers.Flatten(n.X2), n.descriptor)  	
	protoTrain = n.to_proto()

	
	############################# Define the deploy net ###########################

	# We'll use some quick regex to drop the loss layer

	tp_str  = re.split('.*backend: LMDB\n  }\n}', str(protoTrain), maxsplit=2)[2]
	
	protoDep="""name: 'calc'
layer {
    name: "X1"
    type: "Input"
    top: "X1"
    input_param { shape: { dim: 1 dim: 1 dim: 120 dim: 160 } }
}"""

	protoDep += re.split('.*"descriptor"\n}', tp_str)[0] + '  top:  "descriptor"\n}'
	

	return protoTrain, protoDep # string prototext

class CaffeSolver:

	"""
	Caffesolver is a class for creating a solver.prototxt file. It sets default
	values and can export a solver parameter file.
	Note that all parameters are stored as strings. Strings variables are
	stored as strings in strings.
	"""

	def __init__(self, m_iter, trainnet_prototxt_path="proto/train.prototxt", prefix='"calc"', debug=False):

		self.sp = {}

		# critical:
		self.sp['base_lr'] = '0.0009' #'0.0018'
		self.sp['momentum'] = '0.9'

		# looks:
		self.sp['display'] = '25'
		self.sp['snapshot'] = '2500'
		self.sp['snapshot_prefix'] = prefix  # string within a string!

		# learning rate policy
		self.sp['lr_policy'] = '"fixed"'

		# lr_policy param
		self.sp['power'] = '2'

		self.sp['weight_decay'] = '0.0005'
		self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'

		# pretty much never change these.
		self.sp['max_iter'] = m_iter
		self.sp['test_initialization'] = 'false'
		self.sp['average_loss'] = '25'  # this has to do with the display.
		self.sp['iter_size'] = '1'  # this is for accumulating gradients

		if (debug):
			self.sp['max_iter'] = m_iter
			self.sp['display'] = '1'
			self.sp['snapshot'] = '1'


	def add_from_file(self, filepath):
		"""
		Reads a caffe solver prototxt file and updates the Caffesolver
		instance parameters.
		"""
		with open(filepath, 'r') as f:
			for line in f:
				if line[0] == '#':
					continue
				splitLine = line.split(':')
				self.sp[splitLine[0].strip()] = splitLine[1].strip()

	def write(self, filepath):
		"""
		Export solver parameters to INPUT "filepath". Sorted alphabetically.
		"""
		f = open(filepath, 'w')
		for key, value in sorted(self.sp.items()):
			if not(type(value) is str):
				raise TypeError('All solver parameters must be strings')
			f.write('%s: %s\n' % (key, value))
							  

def view_output_size():
	"""
	View the dimmension of the descriptor both flattened into a vector, and in tensor form
	"""
	net = caffe.Net('proto/train.prototxt', caffe.TEST)
	otpt_raw = net.blobs['conv3'].data
	otpt = otpt_raw.copy()
	print 'output matrix shape: ', otpt.shape
	otpt_raw = net.blobs['descriptor'].data
	otpt = otpt_raw.copy()
	print 'output vector shape: ', otpt.shape

def create_net(X1_lmdb, X2_lmdb, max_iter='500000', batch_size=256, prefix='"calc"', data_root="", debugFlag=False):
	"""
	Define, and save the net. Note that max_iter is too high. This is to prevent undertraining. Since we have all the snapshots, we can cross-validate to find the best iteration
	"""
	train_proto, deploy_proto = calc(X1_lmdb, X2_lmdb, data_root=data_root, batch_size=batch_size) 
	if not os.path.isdir("proto"):
		os.makedirs("proto")
	with open('proto/train.prototxt', 'w') as f:
		f.write(str(train_proto))
	with open('proto/deploy.prototxt', 'w') as f:
		f.write(deploy_proto) # deploy_proto is already string, as I could not figure out how to specify Input dimension
	print 'Creating solver ...'
	solver = CaffeSolver(m_iter=max_iter, prefix=prefix, debug=debugFlag)
	solver.write('proto/solver.prototxt') # create the solver proto as well
	view_output_size()
	print 'done'	

def moveModel(model_dir=""):
	"""
	Movve the model and all of its snapshots to specified directory
	"""
	if not os.path.isdir("model/"+model_dir):
		os.makedirs("model/"+model_dir)
	models = glob("*.caffemodel")
	solverstates = glob("*.solverstate")
	for model in models:
		move(model, "model/"+model_dir+"/"+model)
	for solverstate in solverstates:
		move(solverstate, "model/"+model_dir+"/"+solverstate)	

def train(solver_proto_path, snapshot_solver_path=None, init_weights=None, GPU_ID=0):
	"""
	Train the defined net. While we did not use this function for our final net, we used the caffe executable for multi-gpu use, this was used for prototyping
	"""

	import time
	t0 = time.time()
	caffe.set_mode_gpu()
	caffe.set_device(GPU_ID)
	solver = caffe.get_solver(solver_proto_path)
	if snapshot_solver_path is not None:
		solver.solve(snapshot_solver_path) # train from previous solverstate
	else:
		if init_weights is not None:
			solver.net.copy_from(init_weights) # for copying weights from a model without solverstate		
		solver.solve() # train form scratch

	t1 = time.time()
	print 'Total training time: ', t1-t0, ' sec'
	model_dir = "calc_" +  time.strftime("%d-%m-%Y_%I%M%S")
	moveModel(model_dir=model_dir) # move all the model files to a directory  
	print "Moved model to model/"+model_dir


