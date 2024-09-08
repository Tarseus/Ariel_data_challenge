'''
Find breakpoints using derivative and second derivative
'''
from pathlib import Path

import numpy as np
import holoviews as hv
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import os

hv.extension('bokeh')

def get_peaks(gradient_2, inner_shift:int, outter_shift:int)->dict[tuple[int,int], tuple[int, int]]:
    """
    Finds four indices in airs_sig. Two of those mark the transition inside the transiztion zone.
    The other two mark the transition zone just outside of it.
    Parameters
    data: 1D numpy array, the data to be analyzed.
    inner_shift: shift the two indecies in airs_sig identified to exist inside the the transition zone inward, that is to the right 
    (for the start of the transition) and to the left (for the end of the transition) to 
    make the identified transition zone tigher if it needs it. inner_shift's unit is integer index.
    outter_shift: Similar to inner_shift, outter_shift pushes the locations found just outside of the transitions zone outward
    to compensate for inaccurate estimation of transition zone.
    
    """
    gradient_2_slice1 = gradient_2[:len(gradient_2)//2]
    gradient_2_slice2 = gradient_2[len(gradient_2)//2:]
    
    gradient_2_slice1 /= gradient_2_slice1.max()
    
    gradient_2_slice1_neg = -1*gradient_2_slice1
    gradient_2_slice1_neg /= gradient_2_slice1_neg.max()
    
    gradient_2_slice2 /= gradient_2_slice2.max()
    
    gradient_2_slice2_neg = -1*gradient_2_slice2
    gradient_2_slice2_neg /= gradient_2_slice2_neg.max()
    
    in_transition_peaks_slice1, _= find_peaks(gradient_2_slice1, height=0.95)
    out_transition_peaks_slice1, _= find_peaks(gradient_2_slice1_neg, height=0.95)
    
    in_transition_peaks_slice2, _= find_peaks(gradient_2_slice2, height=0.95)
    out_transition_peaks_slice2, _= find_peaks(gradient_2_slice2_neg, height=0.95)
    
    in_transition_peaks_slice2[0] += len(gradient_2)//2
    out_transition_peaks_slice2[0] += len(gradient_2)//2
    
    in_transition_peaks = [in_transition_peaks_slice1[0], in_transition_peaks_slice2[0]]
    out_of_transition_peaks = [out_transition_peaks_slice1[0], out_transition_peaks_slice2[0]]
    
    assert len(in_transition_peaks) == 2, f"Expected 2 in_transition_peaks, got {len(in_transition_peaks)}"
    assert len(out_of_transition_peaks) == 2, f"Expected 2 in_transition_peaks, got {len(out_of_transition_peaks)}"
    return {'inner_peaks': (in_transition_peaks[0]+inner_shift, in_transition_peaks[1]-inner_shift), 
            'outter_peaks': (out_of_transition_peaks[0]-outter_shift, out_of_transition_peaks[1]+outter_shift)}

def find_derivative(data, IDX, train_labels, verbose=False, plot=False):
    data /= np.max(data)
    smoothed_data = savgol_filter(data, 11, 3)
    gradient=np.gradient(smoothed_data)
    gradient/=gradient.max()
    # second derivative
    gradient_2 = np.gradient(gradient)
    gradient_2/=gradient_2.max()
    planet_id = train_labels.index[IDX]

    if verbose:
        print(f"IDX: {IDX}, Planet ID: {planet_id}")
        
    buffer_size = 2
    peaks = get_peaks(gradient_2, buffer_size, buffer_size)
    lower_bounds = [peaks['inner_peaks'][0], peaks['inner_peaks'][1]]
    upper_bounds = [peaks['outter_peaks'][0], peaks['outter_peaks'][1]]
    
    if plot:
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.plot(data, color='#4E79A7', alpha=0.7, label="Data")
        ax1.plot(smoothed_data, label="Smoothed Data", color='#F28E2B')
        ax1.set_title("Data and Smoothed Data")
        ax1.legend()
        
        ax2.plot(gradient, label="First Derivative", color='#E15759')
        ax2.plot(gradient_2, label="Second Derivative", color='#76B7B2')
        ax2.set_title("First and Second Derivatives")
        ax2.legend()
        
        for i in range(2):
            ax1.axvline(x=lower_bounds[i], color="r", linestyle="--")
            ax1.axvline(x=upper_bounds[i], color="r", linestyle="--")
            ax1.axvspan(lower_bounds[i], upper_bounds[i], color="gray", alpha=0.3)
            ax2.axvline(x=lower_bounds[i], color="r", linestyle="--")
            ax2.axvline(x=upper_bounds[i], color="r", linestyle="--")
            ax2.axvspan(lower_bounds[i], upper_bounds[i], color="gray", alpha=0.3)

        fig.tight_layout()
        fig.suptitle(f"Planet ID: {planet_id}, IDX {IDX}", fontsize=16)
        # plt.show()
        os.makedirs('tmp', exist_ok=True)
        plt.savefig(f'tmp/breakpoint_{IDX}.png')
        plt.close()
    
    return upper_bounds[0], lower_bounds[0], lower_bounds[1], upper_bounds[1]