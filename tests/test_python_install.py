import dpc_ann
import numpy as np


query_file = "./data/gaussian_example/gaussian_4_1000.data"
decision_graph_path = "./results/gaussian_4_1000.dg"
output_path = "./results/gaussian_4_1000.cluster"
graph_type = "Vamana"

data = np.load("./data/gaussian_example/gaussian_4_1000.npy").astype("float32")
print(data)
times = dpc_ann.dpc_numpy(
        distance_cutoff=95,
        data=data,
        decision_graph_path=decision_graph_path,
        output_path=output_path,
        graph_type=graph_type,
    )
print(times)


time_reports = dpc_ann.dpc_filenames(
            data_path=query_file,
            decision_graph_path=decision_graph_path,
            output_path=output_path,
            graph_type=graph_type,
            distance_cutoff=95
        )
print(time_reports)
