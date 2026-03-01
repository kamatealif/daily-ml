from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def main() -> None:
    # Create a toy dataset with three natural groups.
    features, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.60,
        random_state=42,
    )

    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = model.fit_predict(features)

    print("Cluster centers:")
    print(model.cluster_centers_)
    print(f"\nInertia: {model.inertia_:.2f}")
    print(f"Silhouette score: {silhouette_score(features, labels):.3f}\n")

    print("First 10 points and assigned cluster:")
    for index, (point, label) in enumerate(zip(features[:10], labels[:10]), start=1):
        print(f"{index:02d}. point={point}, cluster={label}")


if __name__ == "__main__":
    main()
