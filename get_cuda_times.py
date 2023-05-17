import os
import sys
import time
import subprocess
import re
from typing import List, Dict
import pandas as pd


if __name__ == '__main__':
    data_path: str = os.path.join('data')
    out_file: str = 'cuda_times.csv'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(data_path, out_file)):
        os.rename(os.path.join(data_path, out_file), os.path.join(data_path, f'{out_file}_{time.time()}'))

    test_folder: str = os.path.join('polybenchCodesCudaOpenClHMPPOpenAcc', 'CUDA')
    tests: List[str] = ["2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM", "GESUMMV", "MVT", "SYR2K", "SYRK"]
    names: List[str] = ["2DConvolution", "2mm", "3DConvolution", "3mm", "Atax", "Bicg", "Correlation", "Covariance", "Fdtd2d", "Gemm", "Gesummv", "Mvt", "Syr2k", "Syrk"]    
    params: List[int] = [4096, 512, 512, 512, 4096, 8192, 512, 512, 512, 512, 4096, 512, 8192, 512, 512]
    times: Dict[str, List[float]] = dict()
    iterations: int = 10
    pattern = r"GPU Runtime: (\d+\.\d+)s"

    for i, test in enumerate(tests):
        exec_path: str = os.path.join(test_folder, test)
        times[test] = list()
        print(f"Running {exec_path}/{test}.exe")
        for i in range(iterations):
            with open(os.path.join(exec_path, "tmp.txt"), "w") as out:
                process = subprocess.call([os.path.join(exec_path, f"{test}.exe"), f"{params[i]}"], stdout=out)
                process.wait()
            with open(os.path.join(exec_path, "tmp.txt"), "r") as in_file:
                match = re.search(pattern, in_file.read())
                times[test].append(float(match.group(1)))

    print(times)


