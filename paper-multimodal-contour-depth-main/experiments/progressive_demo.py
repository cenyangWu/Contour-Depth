from time import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "..")
from src.data.synthetic_data import main_shape_with_outliers
from src.utils import get_masks_matrix, get_sdfs

from src.visualization import spaghetti_plot

from src.depth.utils import compute_inclusion_matrix, compute_epsilon_inclusion_matrix
from src.depth.inclusion_depth import compute_depths


if __name__ == "__main__":

    #########
    # Setup #
    #########

    outputs_dir = Path("outputs/progressive_demo")
    assert outputs_dir.exists()

    seed_data = 0
    seed_clustering = 0
    seed_init = 0

    N = 300  # fixed
    ROWS = COLS = 512  # fixed
    P_CONTAMINATION = 0.1  # fixed
    BATCH_SIZE = 1
    # CHECKPOINTS = [9, 49, 99]
    CHECKPOINTS = [29, 199, 299]

    rng = np.random.default_rng(seed_data)

    full_masks, full_labs = main_shape_with_outliers(300, ROWS, COLS, 
                                            p_contamination=P_CONTAMINATION, 
                                            return_labels=True, seed=seed_data)
    
    sample_idx = rng.choice(np.arange(300), N, replace=False)
    masks = [full_masks[i] for i in sample_idx]
    labs = full_labs[sample_idx]

    masks_ids = np.arange(N)
    rng.shuffle(masks_ids)
    masks = [masks[i] for i in masks_ids]
    labs = [labs[i] for i in masks_ids]

    ###################
    # Data generation #
    ###################
    
    run_prefix = f"batch-{BATCH_SIZE}"
    path_df = outputs_dir.joinpath(f"{run_prefix}_timings.csv")

    if path_df.exists():
        df = pd.read_csv(path_df, index_col=0)
    else:

        rows = [["seed_data", "step_l", "step_r", "method_id", "time_secs"]]

        saved_checkpoints = 0

        in_counts = [] 
        out_counts = []
        inc_mat = np.zeros((0,0)) # 0 x 0
        current_masks = []

        precompute_in = np.zeros((ROWS, COLS))
        precompute_out = np.zeros((ROWS, COLS))
        
        for step_l in np.arange(N, step=BATCH_SIZE):  # possible to do more than one at a time
            step_r = step_l + BATCH_SIZE
            subset_masks = masks[:step_r]  # current subset we are dealing with 

            in_counts += [0 for _ in range(BATCH_SIZE)]  # add new element to arrays
            out_counts += [0 for _ in range(BATCH_SIZE)]  # add new element to arrays  

            sample_ids = np.arange(step_l, step_r)
            masks_to_add = [masks[s] for s in sample_ids]

            t_start = time()
            for m2a in masks_to_add:
                precompute_in += 1 - m2a
                precompute_out += m2a/m2a.sum()
            t_linear_eid = time() - t_start
            
            t_start = time()            

            # Create new inclusion matrix
            new_inc_mat = np.zeros((inc_mat.shape[0] + BATCH_SIZE, inc_mat.shape[1] + BATCH_SIZE))
            new_inc_mat[0:inc_mat.shape[0], 0:inc_mat.shape[1]] = inc_mat

            # Compute containment of incoming elements in MN'^2 time (if batch is 1 then is O(1))
            new_inc_mat[inc_mat.shape[0]:inc_mat.shape[0]+BATCH_SIZE, inc_mat.shape[1]:inc_mat.shape[1]+BATCH_SIZE] = compute_epsilon_inclusion_matrix(masks_to_add)

            # Compute rows/cols in 2N'N time
            for i, current_mask in enumerate(current_masks):
                for j, mask_to_add in enumerate(masks_to_add):
                    intersect = ((current_mask + mask_to_add) == 2).astype(float)
                    # a_in_b = np.all(current_mask == intersect)
                    # b_in_a = np.all(mask_to_add == intersect)
                    a_in_b = 1-(current_mask * (1-mask_to_add)).sum()/current_mask.sum() #|A-B|/A |(1-B)A) 1 - np.sum(inv_masks & masks[i], axis=(1, 2)) / np.sum(masks[i])
                    b_in_a = 1-(mask_to_add * (1-current_mask)).sum()/mask_to_add.sum()
                    # if a_in_b:
                    #     new_inc_mat[i, j + len(current_masks)] = 1
                    #     in_counts[i] += 1  # progressive quadratic ID
                    #     out_counts[j + len(current_masks)] += 1 # progressive quadratic ID
                    # if b_in_a:
                    #     new_inc_mat[j + len(current_masks), i] = 1
                    #     in_counts[j + len(current_masks)] += 1 # progressive quadratic ID
                    #     out_counts[i] += 1 # progressive quadratic ID

                    new_inc_mat[i, j + len(current_masks)] = a_in_b
                    new_inc_mat[j + len(current_masks), i] = b_in_a
                    in_counts[i] += a_in_b 
                    out_counts[j + len(current_masks)] += a_in_b
                    out_counts[i] += b_in_a
                    in_counts[j + len(current_masks)] += b_in_a
                        

            # Updates for next iteration
            current_masks += masks_to_add
            inc_mat = new_inc_mat.copy()

            t_common = time() - t_start

            # Depth computation

            # progressive/linear
            t_start = time()            
            IN_in = np.array([len(subset_masks) - ((subset_masks[i] / subset_masks[i].sum()) * precompute_in).sum() for i in range(len(subset_masks))])
            IN_out = np.array([len(subset_masks) - ((1-subset_masks[i]) * precompute_out).sum() for i in range(len(subset_masks))])
            d_progressive_leid = np.minimum((IN_in - 1)/len(subset_masks), (IN_out - 1)/len(subset_masks))
            t_linear_eid += time() - t_start

            # - progressive/faster
            # - print(np.array([in_counts, out_counts]).T/len(contours))
            t_start = time()
            d_progressive_f = np.min(np.array([in_counts, out_counts]).T/len(current_masks), axis=1)# compute_depths(subset_masks, inclusion_mat=im, modified=False, fast=False)        
            t_progressive_f = time() - t_start + t_common

            # - progressive/slower
            t_start = time()
            d_progressive = compute_depths(current_masks, inclusion_mat=inc_mat, modified=True, fast=False)
            t_progressive = time() - t_start + t_common
            
            # - Depth calculation (batched)
            t_start = time()        
            batched_im = compute_epsilon_inclusion_matrix(current_masks)
            d_batched = compute_depths(current_masks, inclusion_mat=batched_im, modified=True, fast=False)
            t_batched = time() - t_start + t_common

            rows.append([seed_data, step_l, step_r, "progressive_leid", t_linear_eid])
            rows.append([seed_data, step_l, step_r, "batched_eid", t_batched])
            rows.append([seed_data, step_l, step_r, "progressive_eid", t_progressive])

            print(step_l, step_r, t_batched, t_progressive, t_progressive_f, t_linear_eid)

            print(np.max(d_progressive-d_progressive_f))
            # print(d_progressive_f)
            
            # assert np.all(d_progressive == d_batched)
            # assert np.all(d_progressive == d_progressive_f)
            assert np.all(np.isclose(d_progressive, d_batched))
            assert np.all(np.isclose(d_progressive, d_progressive_f))
            assert np.all(np.isclose(d_progressive, d_progressive_leid))

            if step_l >= CHECKPOINTS[saved_checkpoints] and saved_checkpoints < len(CHECKPOINTS):
                print(f"Saved checkpoint at step_l {step_l}")
                # save depths
                with open(outputs_dir.joinpath(f"{run_prefix}_depths-batched_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(d_batched, f)
                with open(outputs_dir.joinpath(f"{run_prefix}_depths-progressive__eid_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(d_progressive, f)
                with open(outputs_dir.joinpath(f"{run_prefix}_depths-progressive_leid_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(d_progressive, f)
                # save ensemble
                with open(outputs_dir.joinpath(f"{run_prefix}_masks_chkpt-{step_l}.pkl"), "wb") as f:
                    pickle.dump(current_masks, f)
                saved_checkpoints += 1

        df = pd.DataFrame(rows[1:])
        df.columns = rows[0]
        df.to_csv(path_df)  # write to csv

    print(df.head())
    print()

    ############
    # Analysis #
    ############

    import seaborn as sns
    from src.visualization import spaghetti_plot

    CHECKPOINTS = [29, 199, 299]

    timings_df = pd.read_csv(outputs_dir.joinpath("batch-1_timings.csv"))    
    print(timings_df.head())
    timings_df = timings_df.pivot(index=["seed_data", "step_l", "step_r"], columns="method_id", values="time_secs")
    print(timings_df.head())
    timings_df["batched_eid_cumsum"] = timings_df["batched_eid"].cumsum()
    timings_df["progressive_eid_cumsum"] = timings_df["progressive_eid"].cumsum()
    # timings_df["progressive_leid_cumsum"] = timings_df["progressive_leid"].cumsum()
    # timings_df = timings_df.drop(["batched_eid", "progressive_eid", "progressive_leid"], axis=1)
    timings_df = timings_df.drop(["batched_eid", "progressive_eid"], axis=1)
    timings_df = timings_df.reset_index()
    # timings_df = timings_df.melt(["seed_data", "step_l", "step_r"], ["batched_eid_cumsum", "progressive_eid_cumsum", "progressive_leid_cumsum"], value_name="time")
    timings_df = timings_df.melt(["seed_data", "step_l", "step_r"], ["batched_eid_cumsum", "progressive_eid_cumsum"], value_name="time")
    timings_df["method_id"] = timings_df["method_id"].apply(lambda d: dict(batched_eid_cumsum="Batched", progressive_eid_cumsum="Progressive", progressive_leid_cumsum="Progressive (linear eID)")[d]) 
    timings_df = timings_df.rename(lambda d: dict(method_id="Method")[d] if d in ["method_id"] else d, axis=1)

    print(timings_df.head())
    print(timings_df.groupby(["seed_data", "Method"])["time"].min())
    print(timings_df.groupby(["seed_data", "Method"])["time"].max())
    print(timings_df.groupby(["seed_data", "Method"])["time"].mean())
    print(timings_df.groupby(["seed_data", "Method"])["time"].sum())

    # get times 
    


    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")

    sns_plt = sns.lineplot(timings_df, x="step_l", y="time", hue="Method", linewidth=2, ax=ax)
    
    sns_plt.set(xscale="log", yscale="log")
    ax.set_title("Elapsed time vs number of contours processed for \n batched and progressive depth computation")
    ax.set_xlabel("Log(Number of contours displayed)")
    ax.set_ylabel("Log(Elapsed time (seconds))")
    ax.set_xlim(0, 100)

    plt.show()
    
    fig.savefig(outputs_dir.joinpath(f"{run_prefix}_speedcomp.svg"), dpi=300)


    for chkpt in CHECKPOINTS:
        with open(outputs_dir.joinpath(f"{run_prefix}_depths-batched_chkpt-{chkpt}.pkl"), "rb") as f:
            d_batched = pickle.load(f)
        with open(outputs_dir.joinpath(f"{run_prefix}_depths-progressive__eid_chkpt-{chkpt}.pkl"), "rb") as f:
            d_progressive = pickle.load(f)
        # save ensemble
        with open(outputs_dir.joinpath(f"{run_prefix}_masks_chkpt-{chkpt}.pkl"), "rb") as f:
            masks = pickle.load(f)
        assert np.all(d_batched == d_progressive)

        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        spaghetti_plot(masks, iso_value=0.5, arr=d_progressive, is_arr_categorical=False, ax=ax)
        ax.set_axis_off()
        #plt.show()
        fig.savefig(outputs_dir.joinpath(f"{run_prefix}_chkpt-{chkpt}_spaghetti.png"), dpi=300)