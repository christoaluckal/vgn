import collections
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn import io, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
import open3d as o3d
from vgn.grasp import Grasp

MAX_CONSECUTIVE_FAILURES = 2
import random

State = collections.namedtuple("State", ["tsdf", "pc"])

def downsample_to_ratio(pc, target_ratio, tol=0.02, max_iters=20):
    if target_ratio > 1-1e-3:
        return pc, None
    original_size = len(pc.points)
    target_size = int(original_size * target_ratio)

    # Estimate voxel size bounds (adjust as needed)
    min_voxel = 0.0001
    max_voxel = 0.05

    best_voxel = None
    for _ in range(max_iters):
        mid_voxel = (min_voxel + max_voxel) / 2.0
        pc_down = pc.voxel_down_sample(mid_voxel)
        down_size = len(pc_down.points)

        if abs(down_size - target_size) / target_size < tol:
            best_voxel = mid_voxel
            break

        if down_size > target_size:
            min_voxel = mid_voxel
        else:
            max_voxel = mid_voxel

    if best_voxel is None:
        best_voxel = mid_voxel  # Best effort

    return pc.voxel_down_sample(best_voxel), best_voxel

def remove_percent(pc, dir, frac):
    min_bounds = pc.get_min_bound()
    max_bounds = pc.get_max_bound()
    diff = max_bounds - min_bounds
    diff_frac = diff*frac

    curr_points = np.array(pc.points)

    if dir == 'X':
        mask = curr_points[:,0] < diff_frac[0]
    elif dir == 'Y':
        mask = curr_points[:,1] < diff_frac[1]
    elif dir == 'Z':
        mask = curr_points[:,2] < diff_frac[2]
    else:
        raise Exception

    new_points = curr_points[mask]
    new_points = o3d.utility.Vector3dVector(new_points)
    pc.points = new_points
    return pc
    
def add_noise(pc, std):
    curr_points = np.array(pc.points)

    curr_points[:] += np.random.normal(0,std,size=curr_points.shape)
    new_points = o3d.utility.Vector3dVector(curr_points)
    pc.points = new_points
    return pc

def add_random_spheres_to_pcd_fraction(pcd, fraction=0.05, radius=0.02, points_per_sphere=200):

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("The input point cloud has no points.")

    total_sphere_points = int(len(pts) * fraction)
    num_spheres = max(1, total_sphere_points // points_per_sphere)

    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)

    all_points = [pts]

    for _ in range(num_spheres):
        center = np.random.uniform(min_bound, max_bound)

        phi = np.random.uniform(0, np.pi * 2, points_per_sphere)
        costheta = np.random.uniform(-1, 1, points_per_sphere)
        u = np.random.uniform(0, 1, points_per_sphere)

        theta = np.arccos(costheta)
        r = radius * (u ** (1/3))

        xs = r * np.sin(theta) * np.cos(phi) + center[0]
        ys = r * np.sin(theta) * np.sin(phi) + center[1]
        zs = r * np.cos(theta) + center[2]

        sphere_points = np.vstack((xs, ys, zs)).T
        all_points.append(sphere_points)

    combined_points = np.vstack(all_points)

    pcd_with_spheres = o3d.geometry.PointCloud()
    pcd_with_spheres.points = o3d.utility.Vector3dVector(combined_points)

    return pcd_with_spheres

def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    rviz=False,
    downsample_ratio=None,
    coverage_ratio=None,
    noise_std=None,
    artefact_ratio=None,
    artefact_radius=None,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed)
    logger = Logger(logdir, description)

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)
            if downsample_ratio is not None:
                new_pcd, _ = downsample_to_ratio(pc, target_ratio=downsample_ratio)
                print(f"Original size: ", len(np.array(pc.points)))
                print(f"New size: ", len(np.array(new_pcd.points)))
                pc = new_pcd
            elif coverage_ratio is not None:
                if len(coverage_ratio) != 2:
                    raise Exception
                coverage_percent = coverage_ratio[0]
                coverage_dir = coverage_ratio[1]
                new_pcd = remove_percent(pc,coverage_dir,coverage_percent)
                print(f"Original size: ", len(np.array(pc.points)))
                print(f"New size: ", len(np.array(new_pcd.points)))
                pc = new_pcd
            elif noise_std is not None:
                new_pcd = add_noise(pc,noise_std)
                print(f"Original size: ", len(np.array(pc.points)))
                print(f"New size: ", len(np.array(new_pcd.points)))
                pc = new_pcd
            elif artefact_ratio is not None and artefact_radius is not None:
                new_pcd = add_random_spheres_to_pcd_fraction(pc, artefact_ratio, artefact_radius)
                print(f"Original size: ", len(np.array(pc.points)))
                print(f"New size: ", len(np.array(new_pcd.points)))
                pc = new_pcd


            if pc.is_empty():
                print("Empty")
                break  # empty point cloud, abort this round TODO this should not happen

            # visualize scene
            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
                vis.draw_points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, timings["planning"] = grasp_plan_fn(state)

            if len(grasps) == 0:
                print("No grasps")
                for i in range(sim.num_objects):
                    logger.log_grasp(round_id, state, timings, [-100,-100,-100], -int(1e5), Label.FAILURE)
                break  # no detections found, abort this round

            if rviz:
                vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            # execute grasp
            grasp, score = grasps[0], scores[0]
            if rviz:
                vis.draw_grasp(grasp, score, sim.gripper.finger_depth)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label

            print("LOOP:",sim.num_objects, consecutive_failures)


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        if type(grasp) != Grasp:
            qx = -1
            qy = -1
            qz = -1
            qw = -1
            x = grasp[0]
            y = grasp[1]
            z = grasp[2]
            width = -1
        else:
            qx, qy, qz, qw = grasp.pose.rotation.as_quat()
            x, y, z = grasp.pose.translation
            width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
