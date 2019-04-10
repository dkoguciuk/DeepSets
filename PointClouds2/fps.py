import numpy as np

def euclidean_distance(origin, points):
    return ((origin - points)**2).sum(axis=1)

def fps(pointcloud, num_points_final):
    pointcloud_downsampled = np.zeros((num_points_final, 3))
    pointcloud_downsampled[0] = pointcloud[np.random.randint(len(pointcloud))]
    distances = euclidean_distance(pointcloud_downsampled[0], pointcloud)
    for i in range(1, num_points_final):
        pointcloud_downsampled[i] = pointcloud[np.argmax(distances)]
        distances = np.minimum(distances, euclidean_distance(pointcloud_downsampled[i], pointcloud))
    return pointcloud_downsampled
