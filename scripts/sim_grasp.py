import argparse
from pathlib import Path

from vgn.detection import VGN
from vgn.experiments import clutter_removal
import random
from itertools import product

def main(args):

    if args.rviz or str(args.model) == "gpd":
        import rospy

        rospy.init_node("sim_grasp", anonymous=True)

    if str(args.model) == "gpd":
        from vgn.baselines import GPD

        grasp_planner = GPD()
    else:
        grasp_planner = VGN(args.model, rviz=args.rviz)

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

    exp_idx = args.idx

    experiment = exps[exp_idx]
    print(experiment)
    
    dr = None
    cr = None
    ns = None
    art = None
    arr = None
    description = "base"
    if exp_idx < len(resolution_exp) and exp_idx >= 0:
        dr = experiment[0]
        description = f'resolution_{dr}'
    elif exp_idx < len(resolution_exp)+len(coverage_exp) and exp_idx >= 0:
        cr = (experiment[0],experiment[1])
        description = f'coverage_{cr[0]}_{cr[1]}'
    elif exp_idx < len(resolution_exp)+len(coverage_exp)+len(noise_exp) and exp_idx >= 0:
        ns = experiment[0]
        description = f'noise_{ns}'
    elif exp_idx < len(exps) and exp_idx >= 0:
        art = experiment[0]
        arr = experiment[1]
        description = f'artifact_{art}_{arr}'
    else:
        pass

    print("@@@@@@@@@@@",exp_idx,dr,cr,ns,art,arr)

    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=description,
        scene=args.scene,
        object_set=args.object_set,
        num_objects=args.num_objects,
        num_rounds=args.num_rounds,
        seed=ablations["seed"],
        sim_gui=args.sim_gui,
        rviz=args.rviz,
        downsample_ratio=dr,
        coverage_ratio=cr,
        noise_std=ns,
        artefact_ratio=art,
        artefact_radius=arr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--idx", type=int, default=-1)
    args = parser.parse_args()
    main(args)
