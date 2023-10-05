import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OctreeNode:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        self.points = []  # Points within this node
        self.children = []  # Subdivided child nodes
        self.radius = None  # Radius of the node

def calculate_sphere(points):
    # Calculate the sphere that encompasses a list of points
    # Replace this with your sphere calculation method
    # For simplicity, assuming the sphere's center is the average of points and radius is the maximum distance
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    return center, radius

def subdivide_node(node, max_points_per_node):
    if len(node.points) <= max_points_per_node:
        return

    # Calculate the sphere for the current node
    node.center, node.radius = calculate_sphere(node.points)

    # Subdivide into 8 child nodes
    child_size = node.size / 2
    for i in range(8):
        child_center = node.center + np.array([
            (i & 1) * child_size,
            ((i >> 1) & 1) * child_size,
            ((i >> 2) & 1) * child_size
        ])
        child_node = OctreeNode(child_center, child_size)
        node.children.append(child_node)

    # Distribute points to child nodes
    for point in node.points:
        for child_node in node.children:
            if child_node.radius is not None and np.linalg.norm(point - child_node.center) <= child_node.radius:
                child_node.points.append(point)

    # Recursively subdivide child nodes
    for child_node in node.children:
        subdivide_node(child_node, max_points_per_node)

def build_octree(points, max_depth, max_points_per_node):
    root_center = np.mean(points, axis=0)
    root_size = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    root_node = OctreeNode(root_center, root_size / 2)
    root_node.points = points
    subdivide_node(root_node, max_points_per_node)
    return root_node

# Read LAS data from a file
las = laspy.read("C:\\Users\\salci\\PycharmProjects\\pythonProject16\\2743_1234.las")
points = np.vstack((las.x, las.y, las.z)).T

# Example usage:
max_depth = 5  # Maximum depth of the Octree
max_points_per_node = 10  # Maximum points per node before subdivision
root = build_octree(points, max_depth, max_points_per_node)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)

def visualize_octree(node):
    ax.plot([node.center[0]], [node.center[1]], [node.center[2]], marker='o', markersize=5, c='r')
    for child in node.children:
        visualize_octree(child)

visualize_octree(root)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()