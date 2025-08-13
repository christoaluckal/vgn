import argparse
from pathlib import Path

from vgn.detection import VGN
from vgn.experiments import clutter_removal
import random

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
    "resolution_vox_m": [0.20],
    "coverage_frac":    [0.60],
    "noise_sigma_m":    [0.001],
    "artefact_frac":    [0.1],
    "cluster_radius_m": 0.02,
    "seed": 42
    }


    # exp_abl = random.randint(0,3)
    exp_abl = 1
    
    dr = None
    cr = None
    ns = None
    art = None
    arr = None



    if exp_abl == 0:
        dr = random.choice(ablations['resolution_vox_m'])
    elif exp_abl == 1:
        cr = (random.choice(ablations['coverage_frac']),random.choice(['X','Y','Z']))
    elif exp_abl == 2:
        ns = random.choice(ablations['noise_sigma_m'])
    else:
        art = random.choice(ablations['artefact_frac'])
        arr = ablations['cluster_radius_m']

    print(list(ablations.keys())[exp_abl],(dr,cr,ns,art,arr))

    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
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
    args = parser.parse_args()
    main(args)
