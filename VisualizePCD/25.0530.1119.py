#
#
#全体で層を分けてる
#データ量おもたい
#そもそも壁面くりぬいた後に層を分けたい
##

import numpy as np
import matplotlib.pyplot as plt

# --- Align the point cloud to Z-axis (with downsampling if too large) ---
def align_plane_to_axis(points):
    if points.shape[0] > 50000:
        print(f"Too many points ({points.shape[0]}), randomly sampling 50000")
        sampled_idx = np.random.choice(points.shape[0], size=50000, replace=False)
        sample_points = points[sampled_idx]
    else:
        sample_points = points

    centroid = np.mean(sample_points, axis=0)
    centered = sample_points - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]

    target = np.array([0, 0, 1])
    axis = np.cross(normal, target)
    if np.linalg.norm(axis) < 1e-6:
        return points  # Already aligned

    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    rotated = (points - centroid) @ R.T
    return rotated

# --- Visualize Z-layers and histograms ---
def visualize_z_layers_and_hist(points, reflect, n_layers=5, title_prefix="Layer"):
    aligned_points = align_plane_to_axis(points)
    z_values = aligned_points[:, 2]
    min_z, max_z = z_values.min(), z_values.max()
    layer_bounds = np.quantile(z_values, np.linspace(0, 1, n_layers + 1))

    fig = plt.figure(figsize=(15, 3 * n_layers))
    for i in range(n_layers):
        z_min, z_max = layer_bounds[i], layer_bounds[i+1]
        mask = (z_values >= z_min) & (z_values < z_max)
        layer_points = aligned_points[mask]
        layer_reflect = reflect[mask]

        ax_pcd = fig.add_subplot(n_layers, 2, i * 2 + 1, projection='3d')
        ax_pcd.scatter(layer_points[:, 0], layer_points[:, 1], layer_points[:, 2],
                       c=layer_reflect, cmap='hot', s=1)
        ax_pcd.set_title(f"{title_prefix} {i+1}: Z in [{z_min:.2f}, {z_max:.2f}]")
        ax_pcd.set_xlabel("X")
        ax_pcd.set_ylabel("Y")
        ax_pcd.set_zlabel("Z")

        ax_hist = fig.add_subplot(n_layers, 2, i * 2 + 2)
        ax_hist.hist(layer_reflect, bins=100, color='gray')
        ax_hist.set_title(f"Reflectance Histogram {i+1}")
        ax_hist.set_xlabel("Reflectance")
        ax_hist.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# --- Main: read CSV instead of PCD ---
def main():
    # CSVファイルのパス
    csv_path = "mid360_XYZ_Reflect_data_03-28-13-53-11.csv"

    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    if data.shape[1] < 4:
        print("CSV must have at least 4 columns (X, Y, Z, Reflectance)")
        return

    points = data[:, 0:3]
    reflect = data[:, 3]

    visualize_z_layers_and_hist(points, reflect, n_layers=5)

if __name__ == "__main__":
    main()
