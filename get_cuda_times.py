import os
import sys
import time
import subprocess
import re
import statistics
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
    tests: List[str] = ["2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]
    names: List[str] = ["2DConvolution", "2mm", "3DConvolution", "3mm", "Atax", "Bicg", "Correlation", "Covariance", "Fdtd2d", "Gemm", "Gesummv", "Gramschm", "Mvt", "Syr2k", "Syrk"]    
    params: List[int] = [4096, 1024, 512, 512, 4096, 16384, 1024, 1024, 1024, 1024, 16384, 1024, 16384, 1024, 1024]
    times: Dict[str, List[float]] = dict()
    std_devs: List[float] = list()
    means: List[float] = list()
    iterations: int = 10
    pattern = r"GPU Runtime: (\d+\.\d+)s"

    for i, test in enumerate(tests):
        exec_path: str = os.path.join(test_folder, test)
        times[test] = list()
        print(f"Running {exec_path}/{test}.exe")
        for j in range(iterations):
            output = subprocess.check_output([os.path.join(exec_path, f"{test}.exe"), f"{params[i]}"])
            output = output.decode("utf-8")
            match = re.search(pattern, output)
            times[test].append(float(match.group(1)))
    
        std_devs.append(statistics.stdev(times[test]))
        means.append(statistics.mean(times[test]))

    dtimes: Dict = {
        "name": names,
        "size": params,
        "iterations": [10]*len(names),
        "mean_time": means,
        "stddev_time": std_devs,
        "times": list(times.values())
    }

    df: pd.DataFrame = pd.DataFrame(dtimes)
    df.to_csv(os.path.join(data_path, out_file), index=False, header=True)
