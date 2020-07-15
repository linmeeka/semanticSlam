#!/bin/bash
# dirmask=/home/kylin/git/DynaSLAM/masks/walking_rpy_new/mask
# dirresult=/home/kylin/git/DynaSLAM/masks/walking_rpy_new/result
dirmask=$1
dirresult=$2

for file in $dirmask/*; do
    rgb=$(echo $file | sed 's/\(.*\)mask/\1rgb/')
    rgbmask=$(echo $file | sed 's/\(.*\)mask/\1rgbmask/')
    result=$(echo $file | sed 's/\(.*\)mask/\1result/')
    echo $file
    echo $rgb
    python /home/kylin/git/DynaSLAM/CRF/gray2rgb.py $file $rgbmask
    python /home/kylin/git/pydensecrf/examples/inference.py $rgb $rgbmask $result
done

for file in $dirresult/*; do
    maskcrf=$(echo $file | sed 's/result/maskcrf/')
    echo $file
    echo $maskcrf
    python /home/kylin/git/DynaSLAM/CRF/rgb2gray.py $file $maskcrf
done