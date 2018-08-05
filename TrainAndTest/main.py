#!/usr/bin/env python

from os.path import expanduser
from sys import argv

'''
A convenience main for use when writeDatabase and makeNet are compiled into C libraries with cython
'''

if __name__ == '__main__':


        # optional command line interface. Defaults are present, and can be changed above
        import argparse

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='subparser')	

	## makeNet args #################
	net_parser = subparsers.add_parser('net', help='Allows access to functions in makeNet.py, which allow for net definition, training, and descriptor dimension viewing. Run `./main.py net -h` to see all options')

	# Change net defaults here
	default_X1 = 'X1_Places' 
	default_X2 = 'X2_Places'
	default_its = '500000'
	default_batch_sz = 256

	net_parser.add_argument('-x1', dest='X1_lmdb', nargs='?', help='The file name with no path to the database for X1. Default is: '+default_X1, default=default_X1, type=str)

	net_parser.add_argument('-x2', dest='X2_lmdb', nargs='?', help='The file name with full path of the database for X2. Default is: '+default_X2, default=default_X2, type=str)

	net_parser.add_argument('-t', '--train', dest='trainFlag', action='store_true', help='Flag for whether or not to train the proto from the data input', default=False)

	net_parser.add_argument('-s', '--snapshot-path', dest='snapshotProto', nargs='?', help='The file name with full path of the desired solverstate snapshot if you wish to train from a previous model. Default is None, where the model will be trained from scratch', default=None, type=str)

	net_parser.add_argument('-w', '--weights-path', dest='weightsPath', nargs='?', help='Path too .caffemodel weights from a model to init the weights from for training. Only use if solverstate is unavailable', default=None, type=str)

	net_parser.add_argument('-d', '--define', dest='defineFlag', action='store_true', help='Flag for whether or not to define and print the net and solver definition prototxt to $PWD/proto', default=False)

	net_parser.add_argument('-db', '--debug', dest='debugFlag', action='store_true', help='Flag for whether or not to train the net with only 12 iterations for debugging', default=False)

	net_parser.add_argument('-v', '--view-size', dest='viewSizeFlag', action='store_true', help='If true, the descriptor dimensions will be printed', default=False)

	net_parser.add_argument('-i', '--its', dest='max_iter', nargs='?', help='The number of iterations to perform on training, default is: ' + default_its , default=default_its, type=str)        

	net_parser.add_argument('-b', '--batch-sz', dest='batch_size', nargs='?', help='Batch size to train with. Deafult is: ' + str(default_batch_sz), default=default_batch_sz, type=int)        


	## writeDatabase args ###################################################

	db_parser = subparsers.add_parser('db', help='Allows access to functions in writeDatabase.py, which allow for database writing from raw image files, database testing, and optional training when the writing is done (to save yourself a trip). Run `./main.py db -h` to see all options')

	### Change defaults here  ###

	default_outDBNames = ['X1_Places', 'X2_Places']
	# default for dataset to use to create database. Looks in ~/data/Places
	defaultDirNameArr = [expanduser("~") + "/data/Places"]	
	
	db_parser.add_argument('-d1', '--out-db1', dest='outDBName1', nargs='?', help='The directory name with no path of the desired database name for X1. The resulting database will be written to dataRoot/train_data. Default is: '+default_outDBNames[0], default=default_outDBNames[0], type=str)

	db_parser.add_argument('-d2', '--out-db2', dest='outDBName2', nargs='?', help='The directory name with no path of the desired database name for X1. The resulting database will be written to dataRoot/train_data. Default is: '+default_outDBNames[1], default=default_outDBNames[1], type=str)


	db_parser.add_argument('-t', '--test', dest='test', action='store_true', help='Set the flag to test the database by showing images and running unit tests on the image databases', default=False)

	db_parser.add_argument('-d', '--data-dirs', dest='dirNames', nargs='*', help='The space-separated names with full paths of the the directory (directories) containing your dataset(s). Note that if you use glob from the shell, do not use `--data-dirs=<dirs>` or else the literal string will be sent to python. Use `--data-dirs <dirs>`', default=defaultDirNameArr, type=str)

	db_parser.add_argument('-db', '--debug', dest='debugFlag', action='store_true', help='Flag for whether or not to train the net with only 1 iteration for debugging', default=False) 

	db_parser.add_argument('-ta', '--train-after', dest='trainAfter', action='store_true', help='Flag for whether or not to train immediately after witing the database', default=False) 

	db_parser.add_argument('-dr', '--data-root', dest='dataRoot', nargs='?', help='Where you want the train_data directory to be placed. Default is $PWD', default="", type=str)                               
	db_parser.add_argument('-g', '--gpu-id', dest='gpu_id', nargs='?', help='The ID of the gpu to use for training.' , default=0, type=int)        

	db_parser.add_argument('-b', '--batch-sz', dest='batch_size', nargs='?', help='Batch size to train with. Deafult is: ' + str(default_batch_sz), default=default_batch_sz, type=int)        

	## Viewing Warped Images args ###########################################################

	view_parser = subparsers.add_parser('view', help='View an example of the altered image copies')

	view_parser.add_argument('file', metavar='file', help='The file of the image to warp and display', type=str)        
	
	view_parser.add_argument('-s', '--save', dest='save', action='store_true', help='Flag for whether or not to save the images for later use', default=False) 
	
	## Viewing internal representations of a forward pass  ############################

	pass_parser = subparsers.add_parser('pass', help='Viewing internal representations of a forward pass')

	pass_parser.add_argument('file1', metavar='file1', help='The file of the first image to pass', type=str)        
	
	pass_parser.add_argument('file2', metavar='file2', help='The file of the second image to pass', type=str)        
	## Plot text file output from vary-db-size speed tests args ##############################

	plot_parser = subparsers.add_parser('plot', help='Compare varying database times between DBoW2 and DeepLCD')

	plot_parser.add_argument('calc_fl', metavar='calc-fl', help='The file output from ../DeepLCD/build/vary-db-size', type=str)        

	plot_parser.add_argument('-d', '--dbow-fl', dest='dbow_fl', help='The file output from Thirdparty/DBoW2/build/vary-db-size', type=str, default=None)        
	

	## Testing args ########################################################################

	
	test_parser = subparsers.add_parser('test', help='Allows for testing net(s) against some optional benchmark algorithms with precision-recall curves, and timing. Also allows for viewing thresholds used in those curves. Run `./main.py test -h` to see all options')

        test_parser.add_argument('-m', '--model', dest='model_path', nargs='*', help='The file names with full paths of the desired caffemodels for the precision-recall curve.', default="model/calc.caffemodel", type=str)

        test_parser.add_argument('-d', '--data-path', dest='data_path', nargs='?', help='The path to your dataset. Setup with matching images with corresponding filenames in <dataset_path>/live and <dataset_path>/memory', default="test_data/CampusLoopDataset", type=str)

	test_parser.add_argument('-n', '--num-include', dest='num_include', nargs='?', help="Number of images in a sequence to count as a match. Example, if '-n5' is used, Image-7 will count as a match for Image-5 through Image-9 in the database query.", default=7, type=int) 
	
	test_parser.add_argument('-t', '--title', dest='title', nargs='?', help='Plot title', default='Precision-Recall Curve')
	
	test_parser.add_argument('-r', '--resize-net', dest='rz_net', action='store_true', help='If set to true, the net will be resized to the image size. Otherwise, the images will be resized to (120,160)', default=False)

        test_parser.add_argument('-ap', '--alexnet-proto', dest='alex_proto', nargs='?', help='The path to alexnet prototxt definition file', default=None, type=str)

        test_parser.add_argument('-aw', '--alexnet-weights', dest='alex_weights', nargs='?', help='The path to alexnet .caffemodel weights file', default=None, type=str)

	## Parse time! #######################################################################
	
	if len(argv[1:])==0:
		parser.print_help()
		parser.exit()

	args = parser.parse_args()

	if args.subparser=='db':
		from writeDatabase import launch
		launch(160, 120, args.dirNames, [args.outDBName1, args.outDBName2], data_root=args.dataRoot, gpu_id=args.gpu_id, test_db=args.test, debugFlag=args.debugFlag, trainAfter=args.trainAfter, batch_size=args.batch_size)

	elif args.subparser=='view':
		from writeDatabase import showImWarpEx
		showImWarpEx(args.file, args.save)
	
	elif args.subparser=='pass':
		from testNet import view_forward_pass
		view_forward_pass(args.file1, args.file2)

	elif args.subparser=='net':
		from makeNet import view_output_size, train, create_net
		if args.defineFlag:
			# create the net, save the prototext, binary, and solver state
			max_iter = args.max_iter
			if args.debugFlag:
				max_iter = '12'
			create_net(args.X1_lmdb, args.X2_lmdb, max_iter=max_iter, debugFlag=args.debugFlag, batch_size=args.batch_size)

		if args.trainFlag:
			train('proto/solver.prototxt', snapshot_solver_path=args.snapshotProto, init_weights=args.weightsPath)

		if args.viewSizeFlag:
			view_output_size()

	elif args.subparser=='test':
		from testNet import plot
		plot(net_model_path=args.model_path, data_path=args.data_path, num_include=args.num_include, title=args.title, resize_net=args.rz_net, alexnet_proto_path=args.alex_proto, alexnet_weights=args.alex_weights)
	
	elif args.subparser=='plot':
		from testNet import plot_var_db_size
		plot_var_db_size(args.calc_fl, args.dbow_fl)





