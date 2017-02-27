# BEGIN HEADER

set -o errexit

module load CUDA
module load openblas
module load python/3.5.1
module load theano

_workdir=$LSCRATCH
cd $_workdir && pwd
git clone $HOME/IFT6266-project .
ls -l

# END HEADER
