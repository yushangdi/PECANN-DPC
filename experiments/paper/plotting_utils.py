import matplotlib as plt


def set_superplot_font_sizes():
    SMALL_SIZE = 20
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 30

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=MEDIUM_SIZE
    )  # fontsize of the x and y labels for the small plots
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc(
        "figure", labelsize=BIGGER_SIZE
    )  # fontsize of the x and y labels for the big plots


def reset_font_sizes():
    plt.style.use("default")


dataset_name_map = {
    "arxiv-clustering-s2s": "arxiv",
    "reddit-clustering": "reddit",
    "imagenet": "ImageNet",
    "mnist": "MNIST",
    "birds": "birds",
}
