import os
import psutil
import re

def extract_PIDs():
    lines = os.popen('nvidia-smi').read().split('Usage')[-1].split('|\n|')
    pids = []
    gpus = []
    for l in lines:
        x = re.findall(r'\d+', l)
        if len(x) > 1:
            p = psutil.Process(int(x[1]))
            print(f'GPU_{x[0]}',p.cmdline())
            print('\n')

extract_PIDs()
