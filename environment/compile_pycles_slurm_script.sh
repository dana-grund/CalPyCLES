#!/bin/bash

## run as > sbatch compile.sh
## always compiles from scratch!

## requires the PYCLES_PATH.txt stored in this directory
## containing where to find pycles, e.g.
## PYCLES_PATH=/cluster/work/climate/dgrund/git/pressel/pycles/

## JOB SETTINGS
#SBATCH -n 2 # for test
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH -A ls_math
#SBATCH --job-name=compile
#SBATCH --output=compile_pycles.out
#SBATCH --error=compile_pycles.err
##SBATCH --mail-type=END,FAIL
#SBATCH --constraint=ibfabric6 # *
## * submit compilation as job on oldest hardware on Euler to ensure compatibility on all nodes

echo "Compiling PyCLES."
echo Versions used for compilation:
which python
mpiexec --version

# directories
THIS_DIR=$(pwd)
THIS_SCRIPT_DIR=/cluster/work/climate/dgrund/git/dana-grund/CalPyCLES/environment

# where to find PyCLES
cd $THIS_SCRIPT_DIR
PYCLES_PATH=$(grep -o "PYCLES_PATH=[^']*" "PYCLES_PATH.txt" | sed "s/PYCLES_PATH=//")
echo Looking for PyCLES at PYCLES_PATH=$PYCLES_PATH.

# # clean (from scratch)
# echo "Compiling PyCLES from scratch!"
# cd $PYCLES_PATH
# git clean -f -d -X # discard all changes in the .gitignore (compiled files)
# python generate_parameters.py

# compile
cd $PYCLES_PATH
CC=mpicc python setup.py build_ext --inplace

# test
TEST_PATH=$THIS_SCRIPT_DIR/test_pycles_installation
NAMELIST=$TEST_PATH/Straka93_test_nproc2.in
cd $TEST_PATH
echo TEST W/ MPI
mpirun -np 2 python $PYCLES_PATH/main.py $NAMELIST

cd $THIS_DIR
