'''
Find breakpoints using sliding window
'''
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import os

def smooth_data(data, window_size):
    return savgol_filter(data, window_size, 2)  # window size 51, polynomial order 3


def optimize_breakpoint(data, initial_breakpoint, window_size=500, buffer_size=50, smooth_window=121):
    best_breakpoint = initial_breakpoint
    best_score = float("-inf")
    midpoint = len(data) // 2
    smoothed_data = smooth_data(data, smooth_window)
    for i in range(-window_size, window_size):
        new_breakpoint = initial_breakpoint + i
        region1 = data[: new_breakpoint - buffer_size]
        region2 = data[
            new_breakpoint
            + buffer_size : len(data)
            - new_breakpoint
            - buffer_size
        ]
        region3 = data[len(data) - new_breakpoint + buffer_size :]

        # calc on smoothed data
        breakpoint_region1 = smoothed_data[new_breakpoint - buffer_size: new_breakpoint + buffer_size]
        breakpoint_region2 = smoothed_data[-(new_breakpoint + buffer_size): -(new_breakpoint - buffer_size)]

        mean_diff = abs(np.mean(region1) - np.mean(region2)) + abs(
            np.mean(region2) - np.mean(region3)
        )
        var_sum = np.var(region1) + np.var(region2) + np.var(region3)
        range_at_breakpoint1 = (np.max(breakpoint_region1) - np.min(breakpoint_region1))
        range_at_breakpoint2 = (np.max(breakpoint_region2) - np.min(breakpoint_region2))

        mean_range_at_breakpoint = (range_at_breakpoint1 + range_at_breakpoint2) / 2

        score = mean_diff - 0.5 * var_sum + mean_range_at_breakpoint

        if score > best_score:
            best_score = score
            best_breakpoint = new_breakpoint

    return best_breakpoint

def find_window(data, IDX, train_labels, verbose=False):
    # Create a figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    # fig = plt.figure(figsize=(12, 6))
    data /= np.max(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    planet_id = train_labels.index[IDX]

    # for sensor_name in ["AIRS-CH0", "FGS1"]:
    buffer_size = 10
    smooth_window = 31
    window_size = 30
    default_breakpoint = 60

    initial_breakpoint = default_breakpoint

    if verbose:
        print(f"IDX: {IDX}, Planet ID: {planet_id}")
    optimized_breakpoint = optimize_breakpoint(
        data,
        initial_breakpoint,
        window_size=window_size,
        buffer_size=buffer_size,
        smooth_window=smooth_window,
    )

    midpoint = len(data) // 2
    breakpoints = [optimized_breakpoint, 2 * midpoint - optimized_breakpoint]

    # Plot the results
    ax.plot(data, color='#4E79A7', alpha=0.7, label="Original Data")
    ax.plot(smooth_data(data, smooth_window), label="Smoothed Data", color='#F28E2B')

    lower_bounds = []
    upper_bounds = []
    bp1 = breakpoints[0]
    if bp1 - buffer_size < 50:
        lower_bounds.append(50)
    else:
        lower_bounds.append(bp1 - buffer_size)
    if bp1 + buffer_size > 75:
        upper_bounds.append(75)
    else:
        upper_bounds.append(bp1 + buffer_size)

    bp2 = breakpoints[1]
    if bp2 - buffer_size < 115:
        lower_bounds.append(115)
    else:
        lower_bounds.append(bp2 - buffer_size)
    if bp2 + buffer_size > len(data) - 50:
        upper_bounds.append(len(data) - 50)
    else:
        upper_bounds.append(bp2 + buffer_size)
    
    for i in range(2):
        ax.axvline(x=lower_bounds[i], color="r", linestyle="--")
        ax.axvline(x=upper_bounds[i], color="r", linestyle="--")
        ax.axvspan(lower_bounds[i], upper_bounds[i], color="gray", alpha=0.3)

    # ax.set_title(f"{sensor_name}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.suptitle(f"Planet ID: {planet_id}, IDX {IDX}", fontsize=16)
    # plt.show()
    os.makedirs('tmp', exist_ok=True)
    plt.savefig(f'tmp/breakpoint_{IDX}.png')
    plt.close()