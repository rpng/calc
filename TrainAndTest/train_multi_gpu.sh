#!/bin/bash

log_dir=train_logs
mkdir -p $log_dir

caffe_exec=~/caffe/build/tools/caffe

$caffe_exec train --gpu 0,1 --solver proto/solver.prototxt 2>&1 | tee $log_dir/`date +%m-%d-%y`.trainlog

