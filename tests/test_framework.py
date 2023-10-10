import subprocess
import sys
sys.path.append('/home/ubuntu/DPC-ANN')
from post_processors import cluster_eval

def run_command(command):
    try:
        # Run the command and capture the output
        completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        
        # Print the stdout and stderr
        print(f"Command: {command}")
        # print("=== STDOUT ===")
        # print(completed_process.stdout)
        # print("=== STDERR ===")
        # print(completed_process.stderr)
        
    except Exception as e:
        print(f"An error occurred while running the command: {e}")

run_command("./build/dpc_framework_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data \
            --decision_graph_path ./results/gaussian_4_1000.dg --output_file ./results/gaussian_4_1000.cluster \
            --dist_cutoff 8.36")

gt_path = "data/gaussian_example/gaussian_4_1000.gt"
output_path = "results/gaussian_4_1000.cluster"
metrics1 = cluster_eval.eval_clusters_wrapper(gt_path, output_path, verbose=True)
print(metrics1)

run_command("./build/dpc_framework_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data \
            --decision_graph_path ./results/gaussian_4_1000.dg --output_file ./results/gaussian_4_1000.cluster \
            --dist_cutoff 8.36 --density_method Normalized")

gt_path = "data/gaussian_example/gaussian_4_1000.gt"
output_path = "results/gaussian_4_1000.cluster"
metrics1 = cluster_eval.eval_clusters_wrapper(gt_path, output_path, verbose=True)
print(metrics1)