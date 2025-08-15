import subprocess
from itertools import product
ablations = {
        "resolution_vox_m": [0.20, 0.4, 0.5, 0.75],
        "resolution_dir": ['uniform'],
        "coverage_frac":    [0.4, 0.5, 0.6, 0.75],
        "coverage_dir":     ['X', 'Y', 'Z'],
        "noise_sigma_m":    [0.001, 0.005, 0.01],
        "noise_dir": ['uniform'],
        "artefact_frac":    [0.005, 0.001, 0.05, 0.1],
        "artefact_radius": [0.02],
        "seed": 42
    }

resolution_exp = list(product(ablations["resolution_vox_m"],ablations["resolution_dir"]))
coverage_exp = list(product(ablations["coverage_frac"],ablations["coverage_dir"]))
noise_exp = list(product(ablations["noise_sigma_m"],ablations["noise_dir"]))
art_exp = list(product(ablations["artefact_frac"],ablations["artefact_radius"]))

exps = resolution_exp + coverage_exp + noise_exp + art_exp


exp_range = len(exps)
for i in range(-1,exp_range):
    try:
        process = subprocess.run(f"catkin build vgn && python scripts/sim_grasp.py --model gpd --sim-gui --idx {i}", shell=True)
    except Exception as e:
        print(e)
        with open('error.txt','a') as f:
            f.write(f'{i},{e},\n')
