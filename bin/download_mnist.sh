BASEDIR=$(dirname "$0")

mkdir -p $BASEDIR/../data
cd $BASEDIR/..

python3 data_processors/mnist.py