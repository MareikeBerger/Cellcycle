from .CellCycleCombined import CellCycleCombined

from scipy import optimize
import numpy as np

def calculate_cv(crit_frac, parameter_dict, d):
    print(d)
    # make new instance of cell cycle class and run it
    parameter_dict["frac_init"] = crit_frac
    print(parameter_dict)
    myCellCycle = CellCycleCombined(parameter_dict)
    myCellCycle.run_cell_cycle()
    # plottingTools.plot_two_arrays(series_path, myCellCycle.t_init, myCellCycle.v_init_per_ori, r'$t$', r'$v^\ast$', 'init_volume_over_time')
    cv = np.std(myCellCycle.v_init_per_ori[10:]) / np.mean(myCellCycle.v_init_per_ori[10:])
    print('cv was:', cv)
    return cv

def f(x, k, l):
    return (x-k-l)**2

def optimise_golden(a, c, parameter_dict, tol):
    return optimize.golden(calculate_cv, args=(parameter_dict, 0.1), brack=(a, c), tol=tol, maxiter=10)