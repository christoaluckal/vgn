from itertools import product
from random import randint

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

exps = resolution_exp + coverage_exp + noise_exp + art_exp

exp_idx = -1

experiment = exps[exp_idx]

dr = None
cr = None
ns = None
art = None
arr = None

if exp_idx < len(resolution_exp) and exp_idx > 0:
    dr = experiment[0]
elif exp_idx < len(resolution_exp)+len(coverage_exp) and exp_idx > 0:
    cr = (experiment[0],experiment[1])
elif exp_idx < len(resolution_exp)+len(coverage_exp)+len(noise_exp) and exp_idx > 0:
    ns = experiment[0]
elif exp_idx < len(exps) and exp_idx > 0:
    art = experiment[0]
    arr = experiment[1]

print(exps)