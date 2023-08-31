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
sed -i 1,5d s1-label.pa
sed -i 1,5d s2-label.pa
sed -i 1,5d s3-label.pa
sed -i 1,5d s4-label.pa

cd ..
mkdir -p unbalance
cd unbalance
wget wget http://cs.joensuu.fi/sipu/datasets/unbalance-gt-pa.zip
wget http://cs.joensuu.fi/sipu/datasets/unbalance.txt
unzip unbalance-gt-pa.zip
mv unbalance-gt.pa unbalance-label.pa
sed -i 1,4d unbalance-label.pa

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

