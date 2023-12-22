import math
import random


def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def assign_points_to_clusters(centroids, data):
    clusters = [[] for _ in centroids]
    for point in data:
        shortest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(point, centroid)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_centroid_index = i
        clusters[closest_centroid_index].append(point)
    return clusters


def calculate_centroids(clusters):
    return [
        tuple(
            map(
                lambda x: sum(x) / len(cluster_points)
                if len(cluster_points) > 0
                else 0,
                zip(*cluster_points),
            )
        )
        for cluster_points in clusters
    ]


def centroids_have_stabilized(old_centroids, centroids):
    return all(
        euclidean_distance(old, new) < 1e-6
        for old, new in zip(old_centroids, centroids)
    )


def k_means(data, k, max_iterations=100):
    centroids = random.sample(data, k)
    for _ in range(max_iterations):
        old_centroids = centroids
        clusters = assign_points_to_clusters(centroids, data)
        centroids = calculate_centroids(clusters)
        if centroids_have_stabilized(old_centroids, centroids):
            break
    return clusters, centroids
