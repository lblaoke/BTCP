#! /bin/bash

export DISPLAY=':0.0'

#set experimental configuration
CI='python -d -Wonce -Werror'
gammas=(0 1 2)
confs=('conf/test.yaml' 'conf/mnist.yaml')
device='.cache/nvidia-smi-x.xml'
time=$(date +%s)

#clear output and cache
cat /dev/null > ./nohup.out
rm -rf ./.cache

#build folder for this experiment
mkdir -p .cache/pretrained/$time
mkdir -p .cache/trained/$time

#training
for conf in ${confs[@]}
do
	echo ================ pretrain for $conf ================ ;
	nvidia-smi -q -x -f $device ;
	$CI pretrain.py --conf $conf --device $device --time $time || break ;

	for gamma in ${gammas[@]} ;
	do
		echo ================ train gamma = $gamma for $conf ================ ;
		nvidia-smi -q -x -f $device ;
		$CI train.py --gamma $gamma --conf $conf --device $device --time $time ;
	done

	echo -------------------------------------- ;
	echo ---------------- done ---------------- ;
	echo -------------------------------------- ;
done

#evaluating
nvidia-smi -q -x -f $device
for conf in ${confs[@]}
do
	for gamma in ${gammas[@]} ;
	do
		echo ================ evaluate gamma = $gamma for $conf ================ ;
		$CI eval.py --gamma $gamma --conf $conf --device $device --time $time ;
	done

	echo -------------------------------------- ;
	echo ---------------- done ---------------- ;
	echo -------------------------------------- ;
done

#if pause in the end is required, uncomment the next line
#read
