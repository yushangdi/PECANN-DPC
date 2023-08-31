# Get current directory
BASEDIR=$(dirname "$0")

# C++ formatting
clang-format -i $BASEDIR/../*.cpp
clang-format -i $BASEDIR/../*.h
