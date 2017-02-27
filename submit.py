import argparse
import logging
import os
import subprocess

logging.basicConfig(level='INFO', format='[%(name)s] %(levelname)s %(message)s')
log = logging.getLogger('submit.py')

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('--name', required=True)
    argp.add_argument('--script', default='/dev/stdin')
    return argp.parse_args()


def launch(name, script):

    dirp = os.path.dirname(os.path.realpath(__file__))
    
    job_cmd = '{}\n{}\n{}'.format(
        open(dirp + '/submit_header.sh', 'r').read(),
        open(script, 'r').read(),
        open(dirp + '/submit_footer.sh', 'r').read(),
    )

    wdir = os.getenv('HOME')
    
    qsub_cmd = [
        'qsub',
        '-q', '@hades',
        '-N', name,
        '-d', wdir,
        '-joe', '-koe',
        '-l', 'walltime=30:00:00'
    ]

    log.info('command: %s', ' '.join(qsub_cmd))

    proc = subprocess.Popen(
        qsub_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )

    jobid, _ = proc.communicate(job_cmd)

    log.info('Submitted job name=%s, id=%s', name, jobid.strip())

    return jobid


def main():
    args = get_args()
    launch(args.name, args.script)


if __name__ == '__main__':
    main()
