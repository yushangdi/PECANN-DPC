import dpc_ann

query_file = "./data/gaussian_example/gaussian_4_1000.data"
decision_graph_path = "./results/gaussian_4_1000.dg"
output_path = "./results/gaussian_4_1000.cluster"
graph_type = "Vamana"
time_reports = dpc_ann.dpc(
            data_path=query_file,
            decision_graph_path=decision_graph_path,
            output_path=output_path,
            graph_type=graph_type,
            distance_cutoff=95
        )
print(time_reports)
