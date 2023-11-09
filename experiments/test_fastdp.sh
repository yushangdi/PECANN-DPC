#!/bin/bash

python3 test_fastdp.py mnist --window 20 &
python3 test_fastdp.py mnist --window 40 &
python3 test_fastdp.py mnist --window 80 &
python3 test_fastdp.py mnist --window 160 &
python3 test_fastdp.py mnist --window 320 &

wait $(jobs -p)
