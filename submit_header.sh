# BEGIN HEADER

set -o errexit

module load CUDA

_workdir=$LSCRATCH
cd $_workdir && pwd
git clone $HOME/IFT6266-project .
ls -l

export THEANO_FLAGS="floatX=float32,compiledir=$LSCRATCH,device=gpu"

# END HEADER
