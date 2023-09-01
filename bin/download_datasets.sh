#!/bin/bash

# Get current directory
BASEDIR=$(dirname "$0")

mkdir -p $BASEDIR/../data
cd $BASEDIR/../data

mkdir -p s_datasets
cd s_datasets
wget http://cs.joensuu.fi/sipu/datasets/s2.txt
wget http://cs.joensuu.fi/sipu/datasets/s3.txt
wget http://cs.joensuu.fi/sipu/datasets/s-originals.zip
unzip s-originals.zip
mv s1-label.pa s1.gt
mv s2-label.pa s2.gt
mv s3-label.pa s3.gt
mv s4-label.pa s4.gt
sed -i 1,5d s1.gt
sed -i 1,5d s2.gt
sed -i 1,5d s3.gt
sed -i 1,5d s4.gt

cd ..
mkdir -p unbalance
cd unbalance
wget wget http://cs.joensuu.fi/sipu/datasets/unbalance-gt-pa.zip
wget http://cs.joensuu.fi/sipu/datasets/unbalance.txt
unzip unbalance-gt-pa.zip
mv unbalance-gt.pa unbalance.gt
sed -i 1,4d unbalance.gt

cd ..
mkdir -p kdd
cd kdd
wget http://cs.joensuu.fi/sipu/datasets/KDDCUP04Bio.txt
mv KDDCUP04Bio.txt kdd.txt

cd ..
mkdir -p facial
cd facial
wget https://archive.ics.uci.edu/static/public/317/grammatical+facial+expressions.zip
unzip grammatical+facial+expressions.zip -d facial

cd ../..
python3 data_processors/mnist.py
cd data
rm -rf MNIST/raw
rmdir MNIST
mkdir -p mnist
mv mnist.gt mnist/
mv mnist.txt mnist/
