import pandas as pd
import argparse
import glob
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str, help="Folder to read csv files from.")
parser.add_argument("dataset", type=str, help="Dataset to examine.")
parser.add_argument("method", type=str, help="Method to examine.")
args = parser.parse_args()

if args.method in ["Vamana", "HCNNG", "pyNNDescent"]:
    cols = [
        "graph_type",
        "max_degree",
        "alpha",
        "beam_search_construction",
        "beam_search_density",
        "beam_search_clustering",
    ]
if args.method in ["HCNNG", "pyNNDescent"]:
    cols.append("num_clusters")

comparison = "brute force"

csv_files = glob.glob(args.folder + "/*.csv")
df = pd.concat([pd.read_csv(path) for path in csv_files])
df = df[df["dataset"] == args.dataset]
df = df[df["method"].str.contains(args.method)]
df = df[df["comparison"] == comparison]

df[cols] = df["method"].str.split("_", expand=True)
df.drop("method", axis=1, inplace=True)

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


x_pd = df[cols].drop(["graph_type", "alpha"], axis=1)
x = x_pd.to_numpy(dtype="float")
y_acc = df["ARI"].astype("float64")
# y_acc = -np.log(1 - df["ARI"].astype('float64'))
y_time = df["Total time"]

index = ~np.isnan(x).any(axis=1)
x = x[index]
y_acc = y_acc[index]
y_time = y_time[index]


def try_regression(x, y, poly_features):
    if poly_features:
        poly = PolynomialFeatures(2)
        x = poly.fit_transform(x)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    pred_y = regr.predict(x)
    return regr.coef_, r2_score(y, pred_y), mean_absolute_error(y, pred_y)


time_coefs, time_r2, time_mae = try_regression(x, y_time, poly_features=False)

x = np.log(x)
acc_coefs, acc_r2, acc_mae = try_regression(x, y_acc, poly_features=False)

print(f"Time r^2 = {time_r2:.5F}, Accuracy r^2 = {acc_r2:.5F}")
print()
print(
    tabulate(
        zip(x_pd.columns, acc_coefs, time_coefs),
        headers=["Factor", "Accuracy Weight", "Time Weight"],
    )
)
