# BEGIN HEADER

module load CUDA
module load openblas
module load python/3.5.1
module load theano

_workdir=${PBS_JOBID}_${PBS_JOBNAME}
cd $_workdir
cat "${BASH_SOURCE[0]}" > job_script.sh
git clone $HOME/IFT6266-project .

# END HEADER
