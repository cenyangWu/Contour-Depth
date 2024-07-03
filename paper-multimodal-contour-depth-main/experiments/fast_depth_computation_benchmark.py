"""In this file we test the fast depth computation methods for
strict depth formulations.
Learnings have been implemented into the contour_depth library.
"""
from time import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.depth.inclusion_depth import compute_depths as inclusion_depth
from src.depth.band_depth import compute_depths as contour_band_depth
from src.data.synthetic_data import main_shape_with_outliers

if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/fast_depth_computation_benchmark")
    assert outputs_dir.exists()

    depth_fun_dict = dict(cbd=contour_band_depth, id=inclusion_depth, eid=inclusion_depth)

    params = dict(
        cbd = dict(slow=dict(fast=False, modified=False), 
                   fast=dict(fast=True, modified=False)),
        id = dict(slow=dict(fast=False, modified=False), 
                  fast=dict(fast=False, modified=False)),  # strict ID has no fast version
        eid = dict(slow=dict(fast=False, modified=True), 
                   fast=dict(fast=True, modified=True))
    )

    seeds_data = [0, 1, 2, 3, 4]    
    N = 300  # fixed
    ROWS = COLS = 512  # fixed
    P_CONTAMINATION = 0.1  # fixed
    METHOD = ["cbd", "id", "eid"][0]

    ###################
    # Data generation #
    ###################

    # for seed_data in seeds_data:
    #     rng = np.random.default_rng(seed_data)
    #     # build ensemble
    #     masks, labs = main_shape_with_outliers(N, ROWS, COLS, 
    #                                             p_contamination=P_CONTAMINATION, 
    #                                             return_labels=True, seed=seed_data)

    #     path_df = outputs_dir.joinpath(f"{METHOD}_seed{seed_data}_timings.csv")
    #     if path_df.exists():
    #         print("df already exists ...")
    #     else:
    #         # depth computation
    #         rows = [["seed_data", "sample_size", "method_id", "t_slow_secs", "t_fast_secs", "mse"]]
            
    #         sample_idx = np.arange(N)

    #         for sample_size in np.arange(10, N + 1, 10)[::-1]:
    #             print(f"[{METHOD}] Processing {sample_size} ... ")

    #             sample_idx = rng.choice(sample_idx, sample_size, replace=False)
    #             sample_masks = [masks[i] for i in sample_idx]

    #             method_id = METHOD
    #             version_dicts = params[METHOD]

    #             times = dict()
    #             depths = dict()
    #             for version_id, version_params in version_dicts.items():

    #                 if method_id == "cbd" and version_id == "slow" and sample_size >= 150:
    #                     print(f"- METHOD was cbd, sample size {sample_size} is too large, skipping it ...")
    #                     times[version_id] = "NA"
    #                     depths[version_id] = "NA"
    #                 else:
    #                     t_tick = time()
    #                     d = depth_fun_dict[method_id](sample_masks, **version_params)                

    #                     times[version_id] = time() - t_tick
    #                     depths[version_id] = d

    #             try:
    #                 mse = np.mean(np.square(depths['slow'] - depths['fast']))
    #             except:
    #                 mse = "NA"
    #             rows.append([seed_data, sample_size, method_id, times["slow"], times["fast"], mse])

    #             print(rows[-1])
            
    #         df = pd.DataFrame(rows[1:])
    #         df.columns = rows[0]
    #         df.to_csv(path_df)  # write to csv
    
    # print(df.head())
    # print()

    ############
    # Analysis #
    ############

    import seaborn as sns

    seeds_data = [0, 1, 2, 3, 4]    
    METHOD = ["cbd", "id", "eid"]

    dfs = []
    for p in outputs_dir.glob("*timings*"):
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, axis=0)

    print(df.head())


    timings_df = df
    timings_df = timings_df.drop("mse", axis=1)
    timings_df = timings_df.melt(["seed_data", "sample_size", "method_id"], ["t_slow_secs", "t_fast_secs"], "version_id", "time")
    
    timings_df = timings_df.loc[np.negative(np.logical_and(timings_df.method_id=="id", timings_df.version_id=="t_fast_secs")),:]
    # timings_df = timings_df.loc[np.negative(np.logical_and(np.logical_and(timings_df.method_id=="cbd", timings_df.version_id=="t_slow_secs"), timings_df.sample_size>100)),:]

    timings_df["version_id"] = timings_df["version_id"].apply(lambda d: "No" if d == "t_slow_secs" else "Yes") 
    timings_df["method_id"] = timings_df["method_id"].apply(lambda d: dict(cbd="CBD", id="ID", eid="eID")[d]) 
    timings_df = timings_df.rename(lambda d: dict(method_id="Method", version_id="Optimized")[d] if d in ["method_id", "version_id"] else d, axis=1)


    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")

    sns_plt = sns.lineplot(timings_df, 
                           x="sample_size", y="time", 
                           hue="Method", style="Optimized", 
                           linewidth=2, ax=ax)

    # sns_plt.set(xscale="log", yscale="log")
    sns_plt.set(yscale="log")
    ax.set_title("Runtimes vs ensemble size for \n contour band depth and inclusion depth ")
    ax.set_xlabel("Size")
    ax.set_ylabel("Log(Time (seconds))")
    
    plt.show()

    fig.savefig(outputs_dir.joinpath("speed_gains.svg"), dpi=300)
