import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(description='Plot a pareto frontier of total time vs. AMI')
    parser.add_argument('file_path', type=str, help='Path to the CSV results file')
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    plt.figure(figsize=(10, 6))

    # Create a plot to combine all Pareto frontiers
    plt.figure(figsize=(10, 6))
    plt.xlabel('Total time')
    plt.ylabel('ARI')
    plt.title('Combined Pareto Frontier Plot')


    for comparison in ["ground truth", "brute force"]:

        filtered_df = df[df["comparison"] == comparison]

        # Sort the DataFrame by 'Total time' in ascending order and 'ARI' in descending order
        filtered_df = filtered_df.sort_values(by=['Total time', 'ARI'], ascending=[True, False])

        # Initialize variables to track the Pareto frontier
        pareto_frontier = []
        max_ari = float("-inf")

        # Iterate through the sorted DataFrame to find the Pareto frontier
        for _, row in filtered_df.iterrows():
            ari = row['ARI']
            if ari > max_ari:
                max_ari = ari
                pareto_frontier.append(row)

        # Convert the Pareto frontier to a DataFrame
        pareto_df = pd.DataFrame(pareto_frontier)
        plt.plot(pareto_df['Total time'], pareto_df['ARI'], marker='o', linestyle='-', label=f'Pareto Frontier: Comparing against {comparison}')
        
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_frontier_plot.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()