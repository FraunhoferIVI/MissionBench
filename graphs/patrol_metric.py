import json
import math
import os
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt


class PatrolMetric:
    """Compute trajectory metrics and visualizations."""

    def __init__(self, gt_path: Optional[str] = None,
                 vlm_path: Optional[str] = None,
                 vlm_trajectory: Optional[List[Tuple[float, float, float]]] = None):
        """Initialize with file paths to ground truth and VLM trajectories.
            gt_path: Path to ground truth data file (AirSim .txt)
            vlm_path: Path to VLM/predicted data file (JSON)
        """
        self.gt_path = gt_path
        self.vlm_path = vlm_path
        self.gt_trajectory: List[Tuple[float, float, float]] = []
        if vlm_trajectory is not None:
            self.vlm_trajectory = vlm_trajectory
        else:
            self.vlm_trajectory: List[Tuple[float, float, float]] = []
        self.metrics: Dict[str, Any] = {}
        self.validate_files()
        # Load trajectories from files
        if self.gt_path:
            self.gt_trajectory = self._load_trajectory_from_airsim_rec(self.gt_path)
        if self.vlm_path:
            self.vlm_trajectory = self._load_trajectory_from_json(self.vlm_path)
        self.segment_iou = None  # Will be computed in draw_buffer_around_each_point

    def _compute_buffer_geometry(self, buffer_radius: float = 5.0, num_segments: int = 100):
        """Compute buffer geometry for both trajectories.
        
        Args:
            buffer_radius: Radius of buffer zone around each path
            num_segments: Number of segments to resample each path
            
        Returns:
            Dictionary with keys: gt_buffer, vlm_buffer, intersection, union, iou
            Returns None if Shapely is not available
        """
        try:
            from shapely.geometry import LineString
        except ImportError:
            print("Shapely is required for buffer geometry calculation. Please install it.")
            return None

        # Resample paths to 2D segments
        gt_segments = self._resample_trajectory_by_distance(self.gt_trajectory, num_segments)
        vlm_segments = self._resample_trajectory_by_distance(self.vlm_trajectory, num_segments)
        gt_line = LineString([(x, y) for x, y, _ in gt_segments])
        vlm_line = LineString([(x, y) for x, y, _ in vlm_segments])

        # Create buffer polygons
        gt_buffer = gt_line.buffer(buffer_radius)
        vlm_buffer = vlm_line.buffer(buffer_radius)

        # Compute intersection and union
        intersection = gt_buffer.intersection(vlm_buffer)
        union = gt_buffer.union(vlm_buffer)

        # Compute IoU
        iou = intersection.area / union.area if union.area > 0 else 0.0
        
        return {
            'gt_buffer': gt_buffer,
            'vlm_buffer': vlm_buffer,
            'intersection': intersection,
            'union': union,
            'iou': iou
        }

    def compute_actual_iou(self, buffer_radius: float = 5.0, num_segments: int = 100) -> Optional[float]:
        """Compute actual Intersection over Union (IoU) between two paths using buffer zones.
        Draws buffer zones around both paths, creates polygons, and computes IoU based on area.
        Args:
            buffer_radius: Radius of buffer zone around each path
            num_segments: Number of segments to resample each path
        Returns:
            IoU value (area intersection / area union)
        """
        result = self._compute_buffer_geometry(buffer_radius, num_segments)
        if result is None:
            return None
        return result['iou']

    def plot_trajectories_with_buffers(self, buffer_radius: float = 5.0, num_segments: int = 100, 
                                       save_path: Optional[str] = None, 
                                       title: str = "Trajectory Comparison with Buffer Zones"):
        """Plot trajectories with their buffer zones and intersection/union areas.
        
        Args:
            buffer_radius: Radius of buffer zone around each path
            num_segments: Number of segments to resample each path
            save_path: If provided, save figure to this path
            title: Plot title
        """
        # Compute buffer geometry
        geometry = self._compute_buffer_geometry(buffer_radius, num_segments)
        if geometry is None:
            return
        
        gt_buffer = geometry['gt_buffer']
        vlm_buffer = geometry['vlm_buffer']
        intersection = geometry['intersection']
        union = geometry['union']
        iou = geometry['iou']

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot union (light gray)
        if hasattr(union, 'geoms'):  # MultiPolygon
            for geom in union.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, color='lightgray', alpha=0.3, label='Union' if 'Union' not in ax.get_legend_handles_labels()[1] else '')
        else:  # Single Polygon
            x, y = union.exterior.xy
            ax.fill(x, y, color='lightgray', alpha=0.3, label='Union')

        # Create donut-shaped buffers by subtracting the line itself from the buffer
        from shapely.geometry import LineString
        gt_segments = self._resample_trajectory_by_distance(self.gt_trajectory, num_segments)
        vlm_segments = self._resample_trajectory_by_distance(self.vlm_trajectory, num_segments)
        gt_line = LineString([(x, y) for x, y, _ in gt_segments])
        vlm_line = LineString([(x, y) for x, y, _ in vlm_segments])
        
        # Create donut by buffering with a small inner radius
        gt_donut = gt_buffer.difference(gt_line.buffer(0.1))  # Subtract very thin buffer around line
        vlm_donut = vlm_buffer.difference(vlm_line.buffer(0.1))

        # Plot GT buffer donut (green)
        if hasattr(gt_donut, 'geoms'):
            for geom in gt_donut.geoms:
                if hasattr(geom, 'exterior'):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, color='green', alpha=0.2, edgecolor='green', linewidth=1.5, 
                           label='GT Buffer' if 'GT Buffer' not in ax.get_legend_handles_labels()[1] else '')
                    # Fill holes (inner boundaries)
                    for interior in geom.interiors:
                        x, y = interior.xy
                        ax.fill(x, y, color='white', alpha=1.0)
        else:
            if hasattr(gt_donut, 'exterior'):
                x, y = gt_donut.exterior.xy
                ax.fill(x, y, color='green', alpha=0.2, edgecolor='green', linewidth=1.5, label='GT Buffer')
                # Fill holes (inner boundaries)
                for interior in gt_donut.interiors:
                    x, y = interior.xy
                    ax.fill(x, y, color='white', alpha=1.0)

        # Plot VLM buffer donut (red)
        if hasattr(vlm_donut, 'geoms'):
            for geom in vlm_donut.geoms:
                if hasattr(geom, 'exterior'):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, color='red', alpha=0.2, edgecolor='red', linewidth=1.5,
                           label='VLM Buffer' if 'VLM Buffer' not in ax.get_legend_handles_labels()[1] else '')
                    # Fill holes (inner boundaries)
                    for interior in geom.interiors:
                        x, y = interior.xy
                        ax.fill(x, y, color='white', alpha=1.0)
        else:
            if hasattr(vlm_donut, 'exterior'):
                x, y = vlm_donut.exterior.xy
                ax.fill(x, y, color='red', alpha=0.2, edgecolor='red', linewidth=1.5, label='VLM Buffer')
                # Fill holes (inner boundaries)
                for interior in vlm_donut.interiors:
                    x, y = interior.xy
                    ax.fill(x, y, color='white', alpha=1.0)

        # Plot intersection (blue)
        if hasattr(intersection, 'geoms'):
            for geom in intersection.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, color='blue', alpha=0.4, label='Intersection' if 'Intersection' not in ax.get_legend_handles_labels()[1] else '')
        else:
            x, y = intersection.exterior.xy
            ax.fill(x, y, color='blue', alpha=0.4, label='Intersection')

        # Plot actual trajectories
        if self.gt_trajectory:
            gt_x, gt_y, _ = zip(*self.gt_trajectory)
            ax.plot(gt_x, gt_y, 'g-', linewidth=2.5, label='Ground Truth Path', zorder=10)
            ax.scatter(gt_x, gt_y, c='green', s=20, marker='o', zorder=11, edgecolors='white', linewidths=0.5)

        if self.vlm_trajectory:
            vlm_x, vlm_y, _ = zip(*self.vlm_trajectory)
            ax.plot(vlm_x, vlm_y, 'r--', linewidth=2.5, label='VLM Path', zorder=10)
            ax.scatter(vlm_x, vlm_y, c='red', s=20, marker='s', zorder=11, edgecolors='white', linewidths=0.5)

        # Add metrics to title
        full_title = f'{title}\nBuffer Radius={buffer_radius}, IoU={iou:.4f}'
        full_title += f'\nIntersection Area={intersection.area:.2f}, Union Area={union.area:.2f}'
        
        ax.set_title(full_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Buffer visualization saved to {save_path}")
        
        plt.close()
    @staticmethod
    def _load_trajectory_from_airsim_rec(file_path: str) -> List[Tuple[float, float, float]]:
        """Load trajectory from AirSim recording file."""
        trajectory = []
        try:
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  # Skip header
                    items = line.strip().split("\t")
                    x, y, z = float(items[2]), float(items[3]), float(items[4])
                    trajectory.append((x, y, z))
        except Exception as e:
            print(f"⚠ Error loading AirSim trajectory from {file_path}: {e}")
        return trajectory

    @staticmethod
    def _load_trajectory_from_json(file_path: str, drone_pose_key: str = "drone_pose") -> List[Tuple[float, float, float]]:
        """Load trajectory from JSON file."""
        trajectory = []
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    if drone_pose_key in v:
                        pos = v[drone_pose_key]
                        x, y, z = pos["x"], pos["y"], pos["z"]
                        trajectory.append((x, y, z))
        except Exception as e:
            print(f"⚠ Error loading JSON trajectory from {file_path}: {e}")
        print(f"Loaded {len(trajectory)} points from {file_path}")
        return trajectory
    
    def find_nearest_gt_point(self, vx: float, vy: float) -> Optional[Tuple[float, float, float]]:
        """Find the nearest GT point to a given VLM point (vx, vy)."""
        nearest_point, _ = self._find_nearest_point(vx, vy, self.gt_trajectory)
        return nearest_point
    
    @staticmethod
    def _find_nearest_point(px: float, py: float, 
                           trajectory: List[Tuple[float, float, float]]) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        """Find the nearest point in trajectory to (px, py).
        
        Args:
            px, py: Query point coordinates
            trajectory: Trajectory to search
            
        Returns:
            Tuple of (nearest_point, distance). Both None if trajectory is empty.
        """
        if not trajectory:
            return None, None
            
        nearest_point = None
        min_distance = None
        for tx, ty, tz in trajectory:
            dx = px - tx
            dy = py - ty
            distance = math.sqrt(dx * dx + dy * dy)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                nearest_point = (tx, ty, tz)
        return nearest_point, min_distance
    

    def compute_NN_metric(self, source_trajectory: List[Tuple[float, float, float]], 
                          target_trajectory: List[Tuple[float, float, float]]) -> Optional[float]:
        """Compute mean nearest-neighbor distance from source to target.

        Each point in source_trajectory is matched to the closest point in
        target_trajectory by Euclidean distance in 3D. Returns the mean of
        these nearest-neighbor distances. Returns None if either trajectory
        is empty.
        """
        if not source_trajectory or not target_trajectory:
            return None

        distances = []
        for vx, vy, vz in source_trajectory:
            nearest_gt_point = self.find_nearest_gt_point(vx, vy)
            min_dist = math.sqrt((vx - nearest_gt_point[0]) ** 2 + (vy - nearest_gt_point[1]) ** 2) 
            distances.append(min_dist)

        return sum(distances) / len(distances)

    @staticmethod
    def compute_path_length(trajectory: List[Tuple[float, float, float]]) -> float:
        """Compute total path length of a trajectory in 3D.

        Sums Euclidean distances between consecutive points.
        Returns 0.0 if trajectory has fewer than 2 points.
        """
        if len(trajectory) < 2:
            return 0.0

        total = 0.0
        for (x1, y1, z1), (x2, y2, z2) in zip(trajectory[:-1], trajectory[1:]):
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            total += math.sqrt(dx * dx + dy * dy + dz * dz)

        return total

    def plot_trajectories(self, title: str = "Trajectory Comparison", 
                         figsize: Tuple[int, int] = (12, 5),
                         save_path: Optional[str] = None,
                         show_segments: bool = False,
                         num_segments: int = 100):
        """Plot both trajectories in 3D and 2D (top view).
        
        Args:
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: If provided, save figure to this path
            show_segments: If True, show resampled segments instead of all points
            num_segments: Number of segments to show if show_segments=True
        """
        fig = plt.figure(figsize=figsize)
        
        # Resample trajectories if showing segments
        if show_segments:
            gt_plot = self._resample_trajectory_by_distance(self.gt_trajectory, num_segments)
            vlm_plot = self._resample_trajectory_by_distance(self.vlm_trajectory, num_segments)
            gt_label = f'Ground Truth ({num_segments} segments)'
            vlm_label = f'VLM Trajectory ({num_segments} segments)'
            marker_size_gt = 5
            marker_size_vlm = 5
        else:
            gt_plot = self.gt_trajectory
            vlm_plot = self.vlm_trajectory
            gt_label = 'Ground Truth'
            vlm_label = 'VLM Trajectory'
            marker_size_gt = 3
            marker_size_vlm = 3
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        if gt_plot:
            gt_x, gt_y, gt_z = zip(*gt_plot)
            ax1.plot(gt_x, gt_y, gt_z, 'b-o', label=gt_label, markersize=marker_size_gt, linewidth=1.5)
        
        if vlm_plot:
            vlm_x, vlm_y, vlm_z = zip(*vlm_plot)
            ax1.plot(vlm_x, vlm_y, vlm_z, 'r-s', label=vlm_label, markersize=marker_size_vlm, linewidth=1.5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory View')
        ax1.legend()
        ax1.grid(True)
        
        # 2D top view (X-Y plane)
        ax2 = fig.add_subplot(122)
        if gt_plot:
            gt_x, gt_y, _ = zip(*gt_plot)
            ax2.plot(gt_x, gt_y, 'b-o', label=gt_label, markersize=marker_size_gt, linewidth=1.5)
        
        if vlm_plot:
            vlm_x, vlm_y, _ = zip(*vlm_plot)
            ax2.plot(vlm_x, vlm_y, 'r-s', label=vlm_label, markersize=marker_size_vlm, linewidth=1.5)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top View (X-Y Plane)')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close()

    def validate_files(self) -> Dict[str, Any]:
        """Check validity of both trajectory files.
        """
        # Check ground truth file
        if self.gt_path:
            if not os.path.exists(self.gt_path):
                raise FileNotFoundError(f"Ground truth file not found: {self.gt_path}")
            else:
                with open(self.gt_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        raise ValueError("Ground truth file has no data lines")
        else:
            raise ValueError("Ground truth path not provided")
        
        # Check VLM file
        if self.vlm_path:
            if not os.path.exists(self.vlm_path):
                raise FileNotFoundError(f"VLM file not found: {self.vlm_path}")
            else:
                with open(self.vlm_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("JSON root is not a dictionary")
                    else:
                        # Check if any entry has drone_pose
                        has_drone_pose = False
                        for k, v in data.items():
                            if isinstance(v, dict) and "drone_pose" in v:
                                has_drone_pose = True
                                break
                        if not has_drone_pose:
                            raise ValueError("No entries with 'drone_pose' key found")
    
    def draw_buffer_around_each_point(self, buffer_radius: float = 5.0, num_segments: int = 100, save_path: Optional[str] = None):
        """Draw buffer regions around GT path segments (not individual points).
        
        Resamples trajectories into equal-length segments and visualizes coverage.
        VLM segments shown in green if within buffer of GT, red otherwise.
        """
        if self.compute_path_length(self.gt_trajectory) == 0 or self.compute_path_length(self.vlm_trajectory) == 0:
            print("One or both trajectories have zero path length (all points are identical). Cannot compute segment IoU.")
            return 
        # Resample trajectories into equal-length segments
        gt_segments = self._resample_trajectory_by_distance(self.gt_trajectory, num_segments)
        vlm_segments = self._resample_trajectory_by_distance(self.vlm_trajectory, num_segments)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot GT trajectory
        if self.gt_trajectory:
            gt_x, gt_y, _ = zip(*self.gt_trajectory)
            ax.plot(gt_x, gt_y, 'b-', label='Ground Truth Path', linewidth=2, alpha=0.7)
        
        # Draw buffer circles around each GT segment
        for x, y, _ in gt_segments:
            circle = plt.Circle((x, y), buffer_radius, color='blue', alpha=0.15, linewidth=0.5)
            ax.add_patch(circle)
        
        # Plot GT segment points
        if gt_segments:
            gt_seg_x, gt_seg_y, _ = zip(*gt_segments)
            ax.scatter(gt_seg_x, gt_seg_y, c='blue', s=30, marker='o', zorder=5, label='GT Segments')
        
        # Plot VLM trajectory
        if self.vlm_trajectory:
            vlm_x, vlm_y, _ = zip(*self.vlm_trajectory)
            ax.plot(vlm_x, vlm_y, 'gray', linestyle='--', label='VLM Path', linewidth=1.5, alpha=0.5)
        
        # Color VLM segments based on buffer coverage
        green_segs = []
        red_segs = []
        for vx, vy, vz in vlm_segments:
            nearest_gt_point = self.find_nearest_gt_point(vx, vy)
            if nearest_gt_point is not None:
                gx, gy, _ = nearest_gt_point
                dx = vx - gx
                dy = vy - gy
                distance = math.sqrt(dx * dx + dy * dy)
                if distance <= buffer_radius:
                    green_segs.append((vx, vy))
                else:
                    red_segs.append((vx, vy))
        
        # Plot green segments (covered)
        if green_segs:
            green_x, green_y = zip(*green_segs)
            ax.scatter(green_x, green_y, c='green', s=40, marker='s', zorder=6, label='VLM Segments (covered)')
        
        # Plot red segments (not covered)
        if red_segs:
            red_x, red_y = zip(*red_segs)
            ax.scatter(red_x, red_y, c='red', s=40, marker='x', zorder=6, linewidths=2, label='VLM Segments (uncovered)')
        
        # Compute and display metrics
        self.segment_iou = self.compute_path_segment_iou(buffer_radius=buffer_radius)
        
        title = f'Path Segment Coverage (buffer_radius={buffer_radius})\n'
        title += f'Segment IoU={self.segment_iou:.4f}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            print(f"Segment visualization saved as {save_path}")
        plt.close()


    def compute_path_segment_iou(self, buffer_radius: float = 5.0) -> Optional[float]:
        """Compute IoU based on path length coverage.
        
        Discretizes both trajectories into path segments of equal arc-length distance.
        For each GT segment, checks if VLM has a segment within buffer. Returns the
        Jaccard similarity (intersection / union) of covered segments.
        
        Args:
            buffer_radius: Radius of buffer zone
            
        Returns:
            IoU in [0, 1]
            
        Raises:
            ValueError: If either trajectory is empty or has fewer than 2 points, or if path length is 0
        """
        if not self.gt_trajectory:
            raise ValueError("Ground truth trajectory is empty")
        if not self.vlm_trajectory:
            raise ValueError("VLM trajectory is empty")
        if len(self.gt_trajectory) < 2:
            raise ValueError(f"Ground truth trajectory has only {len(self.gt_trajectory)} point(s), need at least 2")
        if len(self.vlm_trajectory) < 2:
            raise ValueError(f"VLM trajectory has only {len(self.vlm_trajectory)} point(s), need at least 2")
        # if all the points in the trajectory are the same, path length will be 0 and IoU is not defined
        if self.compute_path_length(self.gt_trajectory) == 0 or self.compute_path_length(self.vlm_trajectory) == 0:
            return 0.0 
        gt_length = self.compute_path_length(self.gt_trajectory)
        vlm_length = self.compute_path_length(self.vlm_trajectory)
        
        if gt_length == 0:
            raise ValueError("Ground truth trajectory has zero path length (all points are identical)")
        if vlm_length == 0:
            raise ValueError("VLM trajectory has zero path length (all points are identical)")
        
        # Sample points along paths at regular distance intervals
        segment_length = min(gt_length, vlm_length) / 50  # Use 50 segments
        if segment_length == 0:
            return None
        
        gt_segments = self._get_path_segments(self.gt_trajectory, segment_length)
        vlm_segments = self._get_path_segments(self.vlm_trajectory, segment_length)
        
        # Count intersecting segments (within buffer radius using nearest neighbor)
        intersection = 0
        for gx, gy, _ in gt_segments:
            # Find nearest VLM segment to this GT segment using helper method
            _, min_dist = self._find_nearest_point(gx, gy, vlm_segments)
            
            # If nearest neighbor is within buffer, count as intersection
            if min_dist is not None and min_dist <= buffer_radius:
                intersection += 1
        
        # Union is total unique segments (approximation)
        union = len(gt_segments) + len(vlm_segments) - intersection
        
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _resample_trajectory_by_distance(trajectory: List[Tuple[float, float, float]], 
                                         num_points: int) -> List[Tuple[float, float, float]]:
        """Resample trajectory to have approximately num_points evenly spaced by arc length.
        
        Args:
            trajectory: Original trajectory
            num_points: Target number of points
            
        Returns:
            Resampled trajectory with approximately num_points points
            
        Raises:
            ValueError: If trajectory has fewer than 2 points or zero total distance
        """
        if len(trajectory) < 2:
            raise ValueError(f"Trajectory has only {len(trajectory)} point(s), need at least 2 for resampling")
        
        # Compute cumulative distances
        cumulative_distances = [0.0]
        
        for (x1, y1, z1), (x2, y2, z2) in zip(trajectory[:-1], trajectory[1:]):
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            cumulative_distances.append(cumulative_distances[-1] + dist)
        total_distance = cumulative_distances[-1]
        if total_distance == 0:
            raise ValueError("Trajectory has zero total distance (all points are identical)")
        
        # Create target distances evenly spaced
        target_distances = [total_distance * i / (num_points - 1) for i in range(num_points)]
        
        # Interpolate points at target distances
        resampled = []
        traj_idx = 0
        for target_dist in target_distances:
            # Find the two trajectory points that bracket target_dist
            while traj_idx < len(cumulative_distances) - 1 and cumulative_distances[traj_idx + 1] < target_dist:
                traj_idx += 1
            
            if traj_idx >= len(trajectory) - 1:
                resampled.append(trajectory[-1])
            else:
                # Linear interpolation
                d1 = cumulative_distances[traj_idx]
                d2 = cumulative_distances[traj_idx + 1]
                if d2 == d1:
                    resampled.append(trajectory[traj_idx])
                else:
                    t = (target_dist - d1) / (d2 - d1)
                    x1, y1, z1 = trajectory[traj_idx]
                    x2, y2, z2 = trajectory[traj_idx + 1]
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                    resampled.append((x, y, z))
        
        return resampled

    @staticmethod
    def _get_path_segments(trajectory: List[Tuple[float, float, float]], 
                           segment_length: float) -> List[Tuple[float, float, float]]:
        """Get sample points along trajectory at regular arc-length intervals.
        
        Args:
            trajectory: Original trajectory
            segment_length: Distance between samples
            
        Returns:
            List of sample points along the path
        """
        if len(trajectory) < 2:
            return trajectory
        
        segments = [trajectory[0]]
        current_distance = 0.0
        
        for (x1, y1, z1), (x2, y2, z2) in zip(trajectory[:-1], trajectory[1:]):
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            segment_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            
            if segment_dist == 0:
                continue
            
            remaining = segment_dist
            
            while current_distance + remaining >= segment_length:
                t = (segment_length - current_distance) / segment_dist
                next_point = (
                    x1 + t * dx,
                    y1 + t * dy,
                    z1 + t * dz
                )
                segments.append(next_point)
                remaining -= (segment_length - current_distance)
                current_distance = 0.0
            
            current_distance += remaining
        
        segments.append(trajectory[-1])
        return segments


if __name__ == "__main__":
    # Example test cases (update with your actual paths)
    precision_root = "/home/uam/taehyoung/suman/"
    test_cases = [
        # predator laptop paths
        {
            'gt_path': '/home/suman/MT/UAVMissionPlanning/dataset/Patrol/NHEnv/main_road_mission3/airsim_rec.txt',
            'vlm_path': '/home/suman/MT/Experiment_Results/vanilla_step_by_step_approach/gemini-3-flash-preview_experiment_results_11-feb-15.50pm/results/gemini-3-flash-preview/main_road_mission3_11-feb-20.43pm_exp_000032_gemini-3-flash-preview_rep1/debug_outputs/step_by_step_intermediate_results.json',
            'save_plot_path': './Patrol_metric_case2.png',
            'desc': 'NHEnv Main Road Mission 3'
        },
        # precision laptop paths

        {
            'gt_path': '/home/uam/taehyoung/suman/UAVMissionPlanning/dataset/Patrol/NHEnv/main_road_mission3/airsim_rec.txt',
            'vlm_path': '/home/uam/taehyoung/suman/langgraph_ivi/data/results/step_by_step/main_road_mission30/gemini-3-flash-preview_gemini-3-flash-preview_20260211_194400/debug_outputs/step_by_step_intermediate_results.json',
            'save_plot_path': './Patrol_metric_case3.png',
            'desc': 'NHEnv Main Road Mission 3 (Precision Laptop)'
        }
    ]

    buffer_radius = 20.0
    num_segments = 100

    for case in test_cases:
        print(f"\n=== {case['desc']} ===")
        if not os.path.exists(case['gt_path']):
            print(f"Ground truth file not found: {case['gt_path']}. Skipping this case.")
            continue
        if not os.path.exists(case['vlm_path']):
            print(f"VLM file not found: {case['vlm_path']}. Skipping this case.")
            continue
        metric = PatrolMetric(gt_path=case['gt_path'], vlm_path=case['vlm_path'])
        # Check for valid trajectories
        if not metric.gt_trajectory:
            print("Ground truth trajectory is empty. Skipping this case.")
            continue
        if len(metric.gt_trajectory) < 2 or len(metric.vlm_trajectory) < 2:
            print("One or both trajectories have fewer than 2 points. Skipping this case.")
            continue
        if metric.compute_path_length(metric.gt_trajectory) == 0 or metric.compute_path_length(metric.vlm_trajectory) == 0:
            print("One or both trajectories have zero path length (all points are identical). Skipping this case.")
            continue

        try:
            metric.plot_trajectories_with_buffers(buffer_radius=buffer_radius, num_segments=num_segments, save_path=case['save_plot_path'], title=f"{case['desc']} - Buffer Visualization")
            actual_iout = metric.compute_actual_iou(buffer_radius=buffer_radius, num_segments=num_segments)
            print(f"Actual area IoU: {actual_iout:.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Actual area IoU: Error occurred: {e}")

