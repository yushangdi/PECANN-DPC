import numpy as np
import argparse

def eval_cluster_centers_vs_gt(ground_truth_path, decision_graph_path, num_clusters):

    gt = np.loadtxt(ground_truth_path)
    decision_graph = np.loadtxt(decision_graph_path)

    num_to_check = 20
    for i in range(0, num_to_check + 1):
        by_product = np.argsort((num_to_check - i) * np.log(decision_graph[:, 0]) + i * np.log(decision_graph[:, 1]))
        by_product = by_product[::-1]
        top_gts = gt[by_product[:num_clusters]]
        clusters = np.loadtxt("results/imagenet/imagenet_Vamana.cluster")[by_product[:num_clusters]]
        print(top_gts[:5], by_product[:5], clusters[:5])
        print(f"Num different classes, density^{(num_to_check - i) / num_to_check} * dependent_distance^{i / num_to_check}:", len(set(top_gts)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_path", help="Path to the ground truth file.")
    parser.add_argument("decision_graph_path", help="Path to the clustering assignments.")
    parser.add_argument("num_clusters", help="Number of cluster centers to choose.", type=int)
    args = parser.parse_args()
    
    eval_cluster_centers_vs_gt(args.ground_truth_path, args.decision_graph_path, args.num_clusters)