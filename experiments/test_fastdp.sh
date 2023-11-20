#!/bin/bash

python3 test_fastdp.py mnist --window 20 &
python3 test_fastdp.py mnist --window 40 &
python3 test_fastdp.py mnist --window 80 &
python3 test_fastdp.py mnist --window 160 &
python3 test_fastdp.py mnist --window 320 &

python3 test_fastdp.py imagenet --window 20 &
python3 test_fastdp.py imagenet --window 40 &
python3 test_fastdp.py imagenet --window 80 &
python3 test_fastdp.py imagenet --window 160 &
python3 test_fastdp.py imagenet --window 320 &

python3 test_fastdp.py birds --window 20 &
python3 test_fastdp.py birds --window 40 &
python3 test_fastdp.py birds --window 80 &
python3 test_fastdp.py birds --window 160 &
python3 test_fastdp.py birds --window 320 &

python3 test_fastdp.py arxiv-clustering-s2s --window 20 &
python3 test_fastdp.py arxiv-clustering-s2s --window 40 &
python3 test_fastdp.py arxiv-clustering-s2s --window 80 &
python3 test_fastdp.py arxiv-clustering-s2s --window 160 &
python3 test_fastdp.py arxiv-clustering-s2s --window 320 &

python3 test_fastdp.py reddit-clustering --window 20 &
python3 test_fastdp.py reddit-clustering --window 40 &
python3 test_fastdp.py reddit-clustering --window 80 &
python3 test_fastdp.py reddit-clustering --window 160 &
python3 test_fastdp.py reddit-clustering --window 320 &

wait $(jobs -p)
