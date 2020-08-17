import os
import sys

try:
    dataset = sys.argv[1]
    model = sys.argv[2]
    savename = sys.argv[3]
    try:
        k = sys.argv[4]
    except IndexError:
        k = None
    print(f'Running with terminal parameters {dataset} {model} {savename}')
except IndexError:
    print('No parameters are provided')
    sys.exit(1)

logfile = f'logs/{dataset}/{savename}.log'
if k:
    run_command = f'nohup python -u src/runner.py --dataset {dataset} ' + \
                  f'--model {model} --savefile {savename} --k {k} > {logfile} &'
else:
    run_command = f'nohup python -u src/runner.py --dataset {dataset} ' + \
                  f'--model {model} --savefile {savename} > {logfile} &'
os.system(run_command)
os.system('echo ""')
os.system(f'tail -f {logfile}')
