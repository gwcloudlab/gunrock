#!/bin/sh

#SBATCH -o gunrock%j.out
#SBATCH -e gunrock%j.err
# one hour timelimit
#SBATCH --time 7:00:00
# get gpu queue
#SBATCH -p gpu
# need 1 machine
#SBATCH -N 1
# name the job
#SBATCH -J GunrockBeliefPropagation

module load cuda/toolkit
module load libxml2
module load cmake

# build and run gunrock benchmarks
cd ${HOME}/gunrock
cmake . -DCMAKE_BUILD_TYPE=Release
cd ${HOME}/gunrock/bin
make clean && make
rm -f *csv
./bp
