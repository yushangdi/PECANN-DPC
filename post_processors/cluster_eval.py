import numpy as np
from scipy import stats
from collections import Counter
import sklearn
from sklearn import metrics
import os
import json
import sys

def eval_cluster(gt_path, cluster_path):

	print(f'reading gt from {gt_path}')
	print(f'reading result from {cluster_path}')

	with open(gt_path, 'r') as file:
		labels = np.array([int(line.rstrip()) for line in file])

	with open(cluster_path, 'r') as file:
		preds = np.array([int(line.rstrip()) for line in file])


	label_counter = Counter(labels)
	pred_counter = Counter(preds)
	print('groud truth', label_counter)
	print('clustering', pred_counter)
	TP_count = 0
	for label, label_count in label_counter.items():
		ids = np.argwhere(labels == label)[:,0]
		pred, pred_count = stats.mode(preds[ids], axis=None, keepdims=False)
		# print(label, pred)
		if pred_count / (label_count + pred_counter[pred] - pred_count) > 0.5 :
			# print('pass', label, pred)
			TP_count += 1

	recall50 = TP_count / len(label_counter)

	precision50 = TP_count / len(pred_counter)

	AMI = sklearn.metrics.adjusted_mutual_info_score(labels, preds)

	Arand = sklearn.metrics.adjusted_rand_score(labels, preds)

	completeness = sklearn.metrics.completeness_score(labels, preds)

	homogeneity = sklearn.metrics.homogeneity_score(labels, preds)

	result = {}
	result['recall50'] = recall50
	result['precision50'] = precision50
	result['AMI'] = AMI
	result['ARI'] = Arand
	result['completeness'] = completeness
	result['homogeneity'] = homogeneity

	return result

if __name__ == "__main__":
	assert(len(sys.argv) >= 3)
	gt_path = sys.argv[1]
	cluster_path = sys.argv[2]
	print(json.dumps(eval_cluster(gt_path, cluster_path), indent=4))

