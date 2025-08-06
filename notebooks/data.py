# %% [markdown]
# # Data
# 
# Inspect, clean and balance synthetic grasp samples.

# %%
import os
os.chdir('..')

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform

# %%
import rospy
from vgn import vis

rospy.init_node("vgn_vis", anonymous=True)

# %% [markdown]
# Path to the data folder.

# %%
root = Path("data/raw/foo")

# %% [markdown]
# ## Inspection

# %% [markdown]
# Compute the number of positive and negative samples in the dataset.

# %%
df = read_df(root)

# %%
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]

print("Number of samples:", len(df.index))
print("Number of positives:", len(positives.index))
print("Number of negatives:", len(negatives.index))

# %% [markdown]
# Visualize a random sample. Make sure to have a ROS core running and open `config/sim.rviz` in RViz.

# %%
size, intrinsic, _, finger_depth = read_setup(root)

# %%
i = np.random.randint(len(df.index))
scene_id, grasp, label = read_grasp(df, i)
depth_imgs, extrinsics = read_sensor_data(root, scene_id)

tsdf = create_tsdf(size, 120, depth_imgs, intrinsic, extrinsics)
tsdf_grid = tsdf.get_grid()
cloud = tsdf.get_cloud()

vis.clear()
vis.draw_workspace(size)
vis.draw_points(np.asarray(cloud.points))
vis.draw_grasp(grasp, label, finger_depth)

# %% [markdown]
# Plot the distribution of angles between the gravity vector and $Z$ axis of grasps.

# %%
angles = np.empty(len(positives.index))
for i, index in enumerate(positives.index):
    approach = Rotation.from_quat(df.loc[index, "qx":"qw"].to_numpy()).as_matrix()[:,2]
    angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    angles[i] = np.rad2deg(angle)        

# %%
plt.hist(angles, bins=30)
plt.xlabel("Angle [deg]")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## Cleanup
# 
# DANGER: the following lines will modify/delete data.

# %% [markdown]
# Remove grasp positions that lie outside the workspace.

# %%
df = read_df(root)
df.drop(df[df["x"] < 0.02].index, inplace=True)
df.drop(df[df["y"] < 0.02].index, inplace=True)
df.drop(df[df["z"] < 0.02].index, inplace=True)
df.drop(df[df["x"] > 0.28].index, inplace=True)
df.drop(df[df["y"] > 0.28].index, inplace=True)
df.drop(df[df["z"] > 0.28].index, inplace=True)
write_df(df, root)

# %% [markdown]
# Remove unreferenced scenes.

# %%
df = read_df(root)
scenes = df["scene_id"].values
for f in (root / "scenes").iterdir():
    if f.suffix == ".npz" and f.stem not in scenes:
        print("Removed", f)
        f.unlink()

# %% [markdown]
# ## Balance
# 
# Discard a subset of negative samples to balance classes.

# %%
df = read_df(root)

# %%
positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
df = df.drop(i)

# %%
write_df(df, root)

# %%



