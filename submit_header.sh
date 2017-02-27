# BEGIN HEADER

set -o errexit

_workdir=$LSCRATCH
cd $_workdir && pwd
git clone $HOME/IFT6266-project .
ls -l

# END HEADER
