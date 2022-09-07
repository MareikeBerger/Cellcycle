import numpy as np
from scipy.optimize import fsolve
import pandas as pd

from . import PlottingTools as plottingTools
from . import DataStorage as dataStorage
from . import MakeDataframe as makeDataframe

light_blue = (122 / 255, 185 / 255, 218 / 255)
pinkish_red = (247 / 255, 109 / 255, 109 / 255)
dark_blue = (36 / 255, 49 / 255, 94 / 255)
dark_green = (55 / 255, 117 / 255, 80 / 255)

def solveRegulatorConcentrationAnalytically(z, parameters, n_ori, volume, i_series):
    x = z[0]
    F = parameters.basal_rate_regulator[i_series] * (n_ori / volume) / (
            1 + (x / parameters.michaelis_const_regulator[i_series]) ** parameters.hill_coeff_regulator[
        i_series]) - parameters.rate_growth[i_series] * x
    return F


def calculateApproximateRegConc(parameters, n_ori, volume):
    return (parameters.basal_rate_regulator * (n_ori / volume) / (
            parameters.rate_growth *
            parameters.michaelis_const_regulator)) ** (
                   1 / (parameters.hill_coeff_regulator + 1)) * parameters.michaelis_const_regulator


def calculateApproximateInitiatorConc(parameters, regulator_conc, n_ori, volume):
    return parameters.basal_rate_initiator * (n_ori / volume) / \
           (parameters.rate_growth * (1 + (
                   regulator_conc / parameters.michaelis_const_initiator) ** parameters.hill_coeff_initiator))


def calculateApproxInitiatorConcIfHillCoeffsEqual(parameters, regulator_conc):
    return parameters.basal_rate_initiator / parameters.basal_rate_regulator * regulator_conc


def solveActiveFracAnalytically(z, n_ori, volume, parameter_dict):
    x = z[0]
    F = parameter_dict.deactivation_rate_datA / parameter_dict.activation_rate_lipids * x / (parameter_dict.michaelis_const_destr / parameter_dict.total_conc + x) \
        * (parameter_dict.michaelis_const_prod / parameter_dict.total_conc + 1 -x) / (1-x) - volume / n_ori
    return F


def calculate_fixed_point_hill_coeff_2(michaelis_const, growth_rate, basal_rate, n_ori, volume):
    a = michaelis_const * growth_rate / (basal_rate * (n_ori / volume))
    fixed_point = (9 * a ** 2 + np.sqrt(3) * np.sqrt(4 * a ** 6 + 27 * a ** 4)) ** (1 / 3) / (
            2 ** (1 / 3) * 3 ** (2 / 3) * a) \
                  - ((2 / 3) ** (1 / 3) * a) / (9 * a ** 2 + np.sqrt(3) * np.sqrt(4 * a ** 6 + 27 * a ** 4)) ** (1 / 3)
    return fixed_point * michaelis_const


def calculate_v_initi_no_overlap_switch(parameter_set):
    return parameter_set.deactivation_rate_datA / parameter_set.activation_rate_lipids * \
           parameter_set.frac_init * \
           (parameter_set.michaelis_const_prod + (1 - parameter_set.frac_init) *
            parameter_set.total_conc) / \
           ((1 - parameter_set.frac_init) * (
                   parameter_set.michaelis_const_destr + parameter_set.frac_init *
                   parameter_set.total_conc)) - \
           parameter_set.activation_rate_dars1 / parameter_set.activation_rate_lipids


def calculate_v_initi_overlap_switch(parameter_set, i_series=0):
    return (parameter_set.deactivation_rate_rida + parameter_set.deactivation_rate_datA) / \
           parameter_set.activation_rate_lipids * parameter_set.frac_init * \
           (parameter_set.michaelis_const_prod + (1 - parameter_set.frac_init) *
            parameter_set.total_conc) / \
           ((1 - parameter_set.frac_init[i_series]) * (
                   parameter_set.michaelis_const_destr + parameter_set.frac_init *
                   parameter_set.total_conc)) - \
           (parameter_set.activation_rate_dars2 + parameter_set.activation_rate_dars1) / parameter_set.activation_rate_lipids


def plot_time_trace_initiator_fraction(filepath_series, myCellCycle, parameter_dict):
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.active_fraction],
                                          'time',
                                          [r'$V(t)$', r'$n_{ori}(t)$', r'f(t)'], 'initiator_fraction',
                                          r' $\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3))+ r'$ [h^{-1}], f^\ast=$' + str(
                                            parameter_dict.frac_init),
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'],
                                          hlines3=parameter_dict.frac_init)


def plot_time_trace_active_free_initiator_concentration(filepath_series, myCellCycle, parameter_dict):
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.active_fraction * myCellCycle.free_conc],
                                          'time',
                                          [r'$V(t)$', r'$n_{ori}$', r'DnaA-ATP $\%$'], 'active_free_initiator_conc',
                                          r'initiator fraction at $1/\tau_d=$' + str(
                                              np.round(parameter_dict.doubling_rate, 3)),
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'],
                                          hlines3=parameter_dict.critical_free_active_conc)


def plot_time_trace_initiator_fraction_theory(filepath_series, myCellCycle, parameter_dict):
    a = parameter_dict.activation_rate_lipids - parameter_dict.deactivation_rate_datA * myCellCycle.n_ori / myCellCycle.volume
    b = parameter_dict.deactivation_rate_datA * myCellCycle.n_ori / myCellCycle.volume * (
        parameter_dict.michaelis_const_prod + parameter_dict.total_conc) \
        - parameter_dict.activation_rate_lipids * (parameter_dict.total_conc - parameter_dict.michaelis_const_destr)
    c = - parameter_dict.activation_rate_lipids * parameter_dict.total_conc * parameter_dict.michaelis_const_destr

    theory_fraction = ((- b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))/ parameter_dict.total_conc
    print(theory_fraction)
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time, myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.active_fraction, theory_fraction],
                                          'time',
                                          [r'$V(t)$', r'$n_{ori}$', r'DnaA-ATP $\%$'], 'initiator_fraction_th',
                                          r'initiator fraction at $1/\tau_d=$' + str(
                                              np.round(parameter_dict.doubling_rate, 3)),
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'],
                                          legend3=[r'simulations', r'steady state solution'])

def plot_time_trace_active_initiator_concentration(filepath_series, myCellCycle, parameter_dict):
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.active_conc],
                                          'time', [r'$V(t)$', r'$n_{ori}(t)$', r'$[D_{ATP}]$'],
                                          'initiation_concentration', r'$\tau_d^{-1}=$' + str(
                                            np.round(parameter_dict.doubling_rate, 3)) + r'$ [h^{-1}], [D_{\rm ATP}]^\ast=$' + str(
                                            parameter_dict.init_conc) + ' $[\mu m^{-3}]$',
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'])

def plot_time_trace_active_initiator_concentration_with_total(filepath_series, myCellCycle, parameter_dict):
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.total_conc], [myCellCycle.active_conc],
                                          'time', [r'$V(t)$', r'$[D]_{\rm tot}$', r'$[D]_{\rm ATP}$'],
                                          'initiation_concentration_total_conc',
                                          r'initiation concentration at $1/\tau_d=$' + str(
                                              np.round(parameter_dict.doubling_rate, 3)),
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'])


def plot_time_trace_active_initiator_concentration_theory(filepath_series, myCellCycle, parameter_dict):
    a = parameter_dict.activation_rate_lipids - parameter_dict.deactivation_rate_datA * myCellCycle.n_ori / myCellCycle.volume
    b = parameter_dict.deactivation_rate_datA * myCellCycle.n_ori / myCellCycle.volume * (
            parameter_dict.michaelis_const_prod + parameter_dict.total_conc) \
        - parameter_dict.activation_rate_lipids * (parameter_dict.total_conc - parameter_dict.michaelis_const_destr)
    c = - parameter_dict.activation_rate_lipids * parameter_dict.total_conc * parameter_dict.michaelis_const_destr

    theory_conc = ((- b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
    print(theory_conc)
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time, myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.active_conc, theory_conc],
                                          'time', [r'$V(t)$', r'$n_{ori}$', r'$[DnaA-ATP]$'],
                                          'initiation_concentration_th', r'initiation concentration at $1/\tau_d=$' + str(
                                            np.round(parameter_dict.doubling_rate, 3)),
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'],
                                          legend3=[r'simulations', r'steady state solution'])


def lipid_activation_rate(concentrations, parameter_dict):
    return parameter_dict.activation_rate_lipids * (parameter_dict.total_conc - concentrations) / \
                       (parameter_dict.michaelis_const_prod + parameter_dict.total_conc - concentrations)

def complete_activation_rate(concentrations, parameter_dict, time, origin_density):
    if time < parameter_dict.t_doubling_dars2:
        rate_dars2 = parameter_dict.activation_rate_dars2 * origin_density / 2
    else:
        rate_dars2 = parameter_dict.activation_rate_dars2 * origin_density
    if time < parameter_dict.t_doubling_dars1:
        rate_dars1 = parameter_dict.activation_rate_dars1 * origin_density / 2
    else:
        rate_dars1 = parameter_dict.activation_rate_dars1 * origin_density
    return (parameter_dict.activation_rate_lipids + rate_dars1 + rate_dars2) * (parameter_dict.total_conc - concentrations) / \
                       (parameter_dict.michaelis_const_prod + parameter_dict.total_conc - concentrations)

def synthesis_activation_rate(concentrations, parameter_dict):
    print('synth activation rate')
    return parameter_dict.rate_growth * (parameter_dict.total_conc-concentrations)

def dars1_activation_rate(concentrations, parameter_dict, origin_density):
    return parameter_dict.activation_rate_dars1 * origin_density * (parameter_dict.total_conc - concentrations) / \
                       (parameter_dict.michaelis_const_prod + parameter_dict.total_conc - concentrations)


def complete_deactivation_rate(concentrations, parameter_dict, time, origin_density):
    if time < parameter_dict.t_C:
        rate_rida = parameter_dict.deactivation_rate_rida
    else:
        rate_rida = 0
    return (parameter_dict.deactivation_rate_datA + rate_rida) * origin_density * concentrations / (parameter_dict.michaelis_const_destr + concentrations)


def datA_deactivation_rate(concentrations, parameter_dict, origin_density):
    return parameter_dict.deactivation_rate_datA * origin_density * concentrations / (parameter_dict.michaelis_const_destr + concentrations)


def plot_rates_together_simple_switch(myCellCycle, parameter_dict):
    n_time_steps = 6
    concentrations = np.arange(0, parameter_dict.total_conc, 0.01)
    list_concentrations = [concentrations for i in range(n_time_steps)]
    try:
        origin_densities = np.linspace(2/myCellCycle.v_init_per_ori[-1], 1/myCellCycle.v_init_per_ori[-1], n_time_steps)
        list_activation_rates = [lipid_activation_rate(concentrations, parameter_dict) for i in range(n_time_steps)]
        list_deactivation_rates = [datA_deactivation_rate(concentrations, parameter_dict, origin_density) for origin_density in origin_densities]
        plottingTools.plot_series_of_two_arrays(parameter_dict.series_path, list_concentrations, list_activation_rates, list_deactivation_rates,
                                                r'[ATP-DnaA]', r'$\frac{d[D]_{\rm ATP}}{dt}$',
                                                'rate_comparison')
    except:
        print('could not plot_rates_together_simple_switch')


def plot_rates_together_complex_switch(series_path, myCellCycle, parameter_dict):
    n_time_steps = 6
    concentrations = np.arange(0, parameter_dict.total_conc, 0.01)
    list_concentrations = [concentrations for i in range(n_time_steps)]
    list_fractions = [concentrations/parameter_dict.total_conc for i in range(n_time_steps)]
    times = np.linspace(0, n_time_steps, n_time_steps)/n_time_steps / parameter_dict.doubling_rate
    print('times: ', times)
    origin_densities = np.linspace(2/myCellCycle.v_init_per_ori[-1], 1/myCellCycle.v_init_per_ori[-1], n_time_steps)
    list_activation_rates_normalized = [synthesis_activation_rate(concentrations, parameter_dict)/parameter_dict.total_conc for i in range(n_time_steps)]
    # list_activation_rates_normalized = [complete_activation_rate(concentrations, parameter_dict, times[i], origin_densities[i])/parameter_dict.total_conc for i in range(n_time_steps)]
    list_deactivation_rates_normalized = [complete_deactivation_rate(concentrations, parameter_dict, times[i], origin_densities[i])/parameter_dict.total_conc for i in range(n_time_steps)]
    plottingTools.plot_series_of_two_arrays(series_path, list_fractions, list_activation_rates_normalized, list_deactivation_rates_normalized,
                                            r'f', r'$\frac{df}{dt}$',
                                            'rate_comparison_complex', vlines=parameter_dict.frac_init, legend_vlines=r'$f^\ast$')


def plot_rates_together_simple_switch_synthesis(myCellCycle, parameter_dict):
    n_time_steps = 6
    concentrations = np.arange(0, parameter_dict.total_conc, 0.01)
    list_growth_rates =  np.linspace(0.5, 2.5, num=n_time_steps) * np.log(2)
    total_concentrations = parameter_dict.total_conc * np.ones(concentrations.size)
    list_concentrations = [concentrations for i in range(n_time_steps)]
    try:
        origin_densities = np.linspace(2/myCellCycle.v_init_per_ori[-1], 1/myCellCycle.v_init_per_ori[-1], n_time_steps)
        list_activation_rates_with_synth = [lipid_activation_rate(concentrations, parameter_dict) + list_growth_rates[item] * (total_concentrations - list_concentrations[item]) for item in range(n_time_steps)]
        list_activation_rates_without_synth = [lipid_activation_rate(concentrations, parameter_dict) for item in range(n_time_steps)]
        list_deactivation_rates = [datA_deactivation_rate(concentrations, parameter_dict, origin_density) for origin_density in origin_densities]
        # list_growth_rates= [parameter_dict.rate_growth]
        synthesis_at_different_growth_rates = [list_growth_rates[item] * (total_concentrations - list_concentrations[item]) for item in range(len(list_growth_rates))]
        print('synth at different growth rates', synthesis_at_different_growth_rates)
        concentrations_synthesis = [concentrations for i in range(len(list_growth_rates))]
        # list_concentrations.extend(concentrations_synthesis)
        # list_activation_rates.extend(synthesis_at_different_growth_rates)
        # list_deactivation_rates.extend(synthesis_at_different_growth_rates)
        plottingTools.plot_series_of_two_arrays(parameter_dict.series_path, list_concentrations, list_activation_rates_with_synth, list_deactivation_rates,
                                                r'[ATP-DnaA]', r'$\frac{d[D]_{\rm ATP}}{dt}$',
                                                'rate_comparison_synthesis')
        plottingTools.plot_series_of_two_arrays(parameter_dict.series_path, list_concentrations, list_activation_rates_without_synth, list_deactivation_rates,
                                                r'[ATP-DnaA]', r'$\frac{d[D]_{\rm ATP}}{dt}$',
                                                'rate_comparison_without_synthesis')
    except:
        print('could not plot_rates_together_simple_switch_synthesis')


def plot_time_trace_ori_density(filepath_series, myCellCycle, parameter_dict):
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.n_ori / myCellCycle.volume], 'time',
                                          [r'$V(t)$', r'$n_{ori}$', r'$\rho_{ori}$'], 'ori_density',
                                          r'$n_{ori}$ density at $1/\tau_d=$' + str(
                                              np.round(parameter_dict.doubling_rate, 3)),
                                          vlines1=myCellCycle.t_div,
                                          vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue],
                                          legend1=[r'division', r'initiation'])


def plot_time_trace_number_initiators(filepath_series, myCellCycle, parameter_dict):
    title = r'$\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3)) + \
            r'$ [h^{-1}], n^\ast=$' + str(parameter_dict.n_c_max_0) + r'$, K_{D}^{ori}=$' + str(
        parameter_dict.critical_free_conc) + ' $[\mu m^{-3}]$'
    n_regulator = myCellCycle.N_regulator
    # n_regulator[myCellCycle.time>4]=0
    n_initiator = myCellCycle.N_init
    # n_initiator[myCellCycle.time>4]=0
    plottingTools.four_subplots_aligned_n_series(filepath_series, [myCellCycle.time, myCellCycle.time],
                                                 [myCellCycle.volume],
                                                 [myCellCycle.N_init, myCellCycle.sites_total],
                                                 [n_initiator / myCellCycle.volume],
                                                 [myCellCycle.free_conc],
                                                 'time',
                                                 [r'$V(t)$', r'$N_p(t)$', r'$[p](t)$', r'$[p_f](t)$'],
                                                 'number_sites_proteins', title,
                                                 vlines1=myCellCycle.t_div,
                                                 color_vlines1=pinkish_red,
                                                 vlines2=myCellCycle.t_init,
                                                 color_vlines2=dark_blue,
                                                 colors=[dark_green, light_blue],
                                                 legend1=[r'division', r'initiation'],
                                                 legend2=[r'$N_{\rm p}(t)$', r'$N_{\rm t}(t)$'],
                                                 legend3=[r'$[p](t)$', r'$[r](t)$'])


def plot_time_trace_gene_fraction(filepath_series, myCellCycle, parameter_dict):
    title = r'$1/\tau_d=$' + str(np.round(parameter_dict.doubling_rate, 3)) + r', $K_{\rm p}=$' + str(
        parameter_dict.michaelis_const_initiator) + ' nM' + r'$, K_{t}=$' + str(parameter_dict.diss_constant_sites) + ' nM' + \
            r'$, N_{0}=$' + str(parameter_dict.n_c_max_0) + r'$, [p_{\rm f}]^\ast=$' + str(
        parameter_dict.critical_free_conc) + ' nM'
    n_regulator = myCellCycle.N_regulator
    # n_regulator[myCellCycle.time>4]=0
    n_initiator = myCellCycle.N_init
    # n_initiator[myCellCycle.time>4]=0
    plottingTools.four_subplots_aligned_n_series(filepath_series, [myCellCycle.time, myCellCycle.time, myCellCycle.time],
                                                 [myCellCycle.volume],
                                                 [myCellCycle.N_init, myCellCycle.sites_total],
                                                 [n_initiator / myCellCycle.volume,
                                                  n_regulator / myCellCycle.volume],
                                                 [myCellCycle.n_ori / myCellCycle.length_total,
                                                  myCellCycle.n_ori,
                                                  myCellCycle.length_total],
                                                 'time',
                                                 [r'$V(t)$', r'$N_p(t)$', r'$[p](t)$', r'$g_i/\sum_j g_j \, n_s (t)$'],
                                                 'gene_fraction', title,
                                                 vlines1=myCellCycle.t_div,
                                                 color_vlines1=pinkish_red,
                                                 vlines2=myCellCycle.t_init,
                                                 color_vlines2=dark_blue,
                                                 colors=[dark_green, light_blue, pinkish_red],
                                                 legend1=[r'division', r'initiation'],
                                                 legend2=[r'$N_{\rm p}(t)$', r'$N_{\rm t}(t)$'],
                                                 legend3=[r'$[p](t)$', r'$[r](t)$'])

def plot_time_trace_lipids(filepath_series, myCellCycle, parameter_dict):
    if parameter_dict.version_of_lipid_regulation == 'proteome_sector':
        lipid_conc = myCellCycle.lipid_conc
    else:
        lipid_conc = myCellCycle.N_lipids / myCellCycle.volume
    #myCellCycle.N_regulator_lipids / myCellCycle.volume
    title = r'$1/\tau_d=$' + str(np.round(parameter_dict.doubling_rate, 3)) + r', $K_{\rm p}=$' + str(
        parameter_dict.michaelis_const_initiator) + ' nM' + r'$, N_{0}=$' + str(parameter_dict.n_c_max_0) + r'$, \langle l \rangle=$' + str(np.round(np.mean(lipid_conc), 3)) \
            + r'$, CV_l=$' + str(np.round(np.std(lipid_conc)/np.mean(lipid_conc), 3))
    plottingTools.four_subplots_aligned_n_series(filepath_series, [myCellCycle.time, myCellCycle.time],
                                                 [myCellCycle.volume],
                                                 [myCellCycle.N_lipids, myCellCycle.N_regulator_lipids],
                                                 [lipid_conc],
                                                 [myCellCycle.active_fraction],
                                                 'time',
                                                 [r'$V(t)$', r'$N_l(t)$', r'$[l](t)$', r'$f$'],
                                                 'number_lipids', title,
                                                 vlines1=myCellCycle.t_div,
                                                 color_vlines1=pinkish_red,
                                                 vlines2=myCellCycle.t_init,
                                                 color_vlines2=dark_blue,
                                                 colors=[dark_green, light_blue],
                                                 legend1=[r'division', r'initiation'],
                                                 legend2=[r'$N_{\rm l}(t)$', r'$N_{\rm r}(t)$'],
                                                 legend3=[r'$[l](t)$', r'$[r](t)$'])

def plot_time_trace_switch_titration_combined(filepath_series, myCellCycle, parameter_dict):
    title = r'$\tau_d^{-1}=$' + str(np.round(parameter_dict.doubling_rate, 3)) + \
            r'$ [h^{-1}], n^\ast=$' + str(parameter_dict.n_c_max_0) + r'$, [D_{\rm ATP, f}]^\ast=$' + str(
        parameter_dict.critical_free_active_conc) + ' $[\mu m^{-3}]$'
    plottingTools.four_subplots_aligned_n_series(filepath_series, [myCellCycle.time],
                                                 [myCellCycle.volume],
                                                 [myCellCycle.free_conc],
                                                 [myCellCycle.active_fraction],
                                                 [myCellCycle.active_fraction * myCellCycle.free_conc],
                                                 'time',
                                                 [r'$V(t)$', r'$[D_{\rm ATP, f}]$', r'$f(t)$', r'$[D]_{\rm ATP, f}(t)$'],
                                                 'active_fraction_and_active_free_fraction', title,
                                                 vlines1=myCellCycle.t_div,
                                                 color_vlines1=pinkish_red,
                                                 vlines2=myCellCycle.t_init,
                                                 color_vlines2=dark_blue,
                                                 colors=[dark_green, light_blue],
                                                 legend1=[r'division', r'initiation'])


def plot_time_trace_concentration_initiators(filepath_series, myCellCycle, parameter_dict):
    title = r'$1/\tau_d=$' + str(np.round(parameter_dict.doubling_rate, 3)) + r', $K_{\rm p}=$' + str(
        parameter_dict.michaelis_const_initiator) + ' nM' + r'$, K_{t}=$' + str(parameter_dict.diss_constant_sites) + ' nM' + \
            r'$, N_{0}=$' + str(parameter_dict.n_c_max_0) + r'$, [p_{\rm f}]^\ast=$' + str(
        parameter_dict.critical_free_conc) + ' nM'
    fixed_point_h2 =  calculate_fixed_point_hill_coeff_2(parameter_dict.michaelis_const_regulator, parameter_dict.rate_growth, parameter_dict.basal_rate_regulator, myCellCycle.n_ori, myCellCycle.volume)
    fixed_point = fixed_point_h2
    plottingTools.plot_series_of_one_array(filepath_series, [myCellCycle.time, myCellCycle.time, myCellCycle.time],
                                                 [myCellCycle.N_init/myCellCycle.volume,fixed_point_h2, fixed_point],
                                                 'time', r'[p]', 'concentration_with_fixed_point',labels=[r'[p]', r'$[\rm p^\ast_{h2}]$', r'$[\rm p^\ast]$'])
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time, myCellCycle.time], [myCellCycle.volume],
                                          [myCellCycle.n_ori], [myCellCycle.N_init/myCellCycle.volume,fixed_point_h2], r'$t$',
                                          [r'$V(t)$', r'$n_{ori}(t)$', r'$[p(t)]$'], 'concentration_with_fixed_point',
                                          r'$n_{ori}$ density at $\lambda=$' + str(
                                              np.round(parameter_dict.rate_growth, 3)), legend3=[r'[p]', r'$[\rm p^\ast]$'])

def plot_time_trace_activation_inactivation_rates(filepath_series, myCellCycle, parameter_dict):
    if (parameter_dict.activation_rate_dars2 == 0):
        dars2_rates = myCellCycle.activation_rate_dars2_tot
    else:
        dars2_rates = myCellCycle.activation_rate_dars2_tot / parameter_dict.activation_rate_dars2
    if (parameter_dict.activation_rate_dars1 == 0):
        dars1_rates = myCellCycle.activation_rate_dars1_tot
    else:
        dars1_rates = myCellCycle.activation_rate_dars1_tot / parameter_dict.activation_rate_dars1
    plottingTools.three_subplots_n_series(filepath_series, [myCellCycle.time, myCellCycle.time, myCellCycle.time],
                                          [myCellCycle.active_fraction],
                                          [myCellCycle.activation_rate_lipids_tot/parameter_dict.activation_rate_lipids,
                                           dars2_rates,
                                           dars1_rates],
                                          [myCellCycle.destr_rate_datA_tot/parameter_dict.deactivation_rate_datA,
                                           myCellCycle.destr_rate_rida_tot/parameter_dict.deactivation_rate_rida],
                                          'time',
                                          [r'$f^\ast$', r'activation rates $\alpha$', r'deactivation rates $\beta$'],
                                          'act_deact_rates',
                                          r'production and destruction rates at $1/\tau_d=$' + str(
                                              np.round(parameter_dict.doubling_rate, 3)),
                                          legend2=[r'$\alpha_l/\alpha_{l0}$', r'$\alpha_{d2}/\alpha_{d20}$', r'$\alpha_{d1}/\alpha_{d1}$'],
                                          legend3=[r'$\beta_{datA}/\beta_{datA}$', r'$\beta_{Rida}/\beta_{Rida0}$'],
                                          vlines1=myCellCycle.t_div, vlines2=myCellCycle.t_init,
                                          color_vlines1=pinkish_red,
                                          color_vlines2=dark_blue,
                                          colors=[dark_green, light_blue, 'r'],
                                          legend1=[r'division', r'initiation'])

def return_array_of_one_variable_for_nseries(parameters, name_dataset, variable):
    array_of_variable = np.array([])
    for i_series in range(parameters.n_series):
        # file name and path for series:
        filename_series = parameters.series_name + '_' + str(i_series)
        filepath_series = parameters.file_path + '/' + filename_series
        filename_dataset = 'dataset_'+ str(i_series)
        data_frame = dataStorage.openDataFrameHdf5(filepath_series, filename_dataset, key=name_dataset)
        array_of_variable = np.append(array_of_variable, data_frame.iloc[-1][variable])
    return array_of_variable

def return_list_of_time_arrays_for_nseries(parameters, name_dataset, variable):
    list_of_variables = []
    for i_series in range(parameters.n_series):
        # file name and path for series:
        filename_series = parameters.file_name + '_' + str(i_series)
        filename_series = parameters.file_name + '_' + str(i_series)
        filepath_series = parameters.file_path + '/' + filename_series
        data_frame = dataStorage.openDataFrameHdf5(filepath_series, key=name_dataset)
        list_of_variables.append(data_frame[variable])
    return list_of_variables

def plot_initiation_volume_for_different_growth_rates_switch_theory(parameters):
    v_init_th_non_overlap = calculate_v_initi_no_overlap_switch(parameters)
    v_init_th_overlap = calculate_v_initi_overlap_switch(parameters)
    series_init_volumes = return_array_of_one_variable_for_nseries(parameters, 'dataset_init_events', 'v_init')
    series_init_n_oris = return_array_of_one_variable_for_nseries(parameters, 'dataset_init_events', 'n_ori_init')
    series_v_init_per_ori = series_init_volumes/series_init_n_oris
    plottingTools.plot_two_arrays(parameters.file_path, parameters.doubling_rate, series_v_init_per_ori,
                                  r'$1/\tau_d$', r'$V^\ast$', 'initiation_volume', cv=1,
                                  theory=[v_init_th_non_overlap, v_init_th_overlap],
                                  label_th=[r'$V^\ast_{{th,NO}}={}$'.format(np.round(v_init_th_non_overlap, 3)), r'$V^\ast_{{th,O}}={}$'.format(np.round(v_init_th_overlap, 3))])

def plot_birth_volume_for_different_growth_rates(parameters):
    v_init_th_non_overlap = calculate_v_initi_no_overlap_switch(parameters)
    theory_birth_vol = v_init_th_non_overlap * 2 ** (parameters.t_CD * parameters.doubling_rate) / 2
    series_birth_volumes = return_array_of_one_variable_for_nseries(parameters, 'dataset_div_events', 'v_b')
    plottingTools.plot_series_of_one_array(parameters.file_path, [parameters.doubling_rate, parameters.doubling_rate], [series_birth_volumes, theory_birth_vol],
                                   r'$1/\tau_d$', r'$V_b$', 'birth_volume_theory', labels=['from evolution of model', r'$V_b=\frac{V^\ast}{2} \times 2^{\tau_{cc}/\tau_d}$'])

def plot_minimal_fraction_for_different_growth_rates(parameter_dict):
    active_concentration = return_list_of_time_arrays_for_nseries(parameter_dict, 'dataset_time_traces', 'active_conc')
    series_min_frac = [np.min(active_concentration[item][int(len(active_concentration[item])/2):-1]) for item in range(len(active_concentration))]
    plottingTools.plot_two_arrays(parameter_dict.file_path, parameter_dict.doubling_rate, np.array(series_min_frac),
                                  r'$1/\tau_d$', r'$f_{min}$', 'minimal_active_frac')


def plot_conc_and_initiation_volume_for_different_growth_rates_titration_theory(parameter_set):
    x_axis = parameter_set.doubling_rate

    # series_init_volumes = return_array_of_one_variable_for_nseries(parameters, 'dataset_init_events', 'v_init')
    # series_init_n_oris = return_array_of_one_variable_for_nseries(parameters, 'dataset_init_events', 'n_ori_init')
    # series_v_init_per_ori = series_init_volumes/series_init_n_oris
    series_init_volumes = return_list_of_time_arrays_for_nseries(parameter_set, 'dataset_init_events', 'v_init')
    series_init_n_oris = return_list_of_time_arrays_for_nseries(parameter_set, 'dataset_init_events', 'n_ori_init')
    print(series_init_volumes, series_init_n_oris)
    series_init_v_average = np.array([np.mean(series_init_volumes[item]) for item in
                       range(len(series_init_volumes))])
    series_init_n_oris_average = np.array(
        [np.mean(series_init_n_oris[item]) for item in
         range(len(series_init_n_oris))])
    series_v_init_per_ori = series_init_v_average/series_init_n_oris_average

    series_n_oris = return_list_of_time_arrays_for_nseries(parameter_set, 'dataset_time_traces', 'n_ori')
    series_n_initiator = return_list_of_time_arrays_for_nseries(parameter_set, 'dataset_time_traces', 'N_init')
    series_volume = return_list_of_time_arrays_for_nseries(parameter_set, 'dataset_time_traces', 'volume')

    series_average_volume = np.array([np.mean(series_volume[item][int(len(series_volume[item]) / 2):-1]) for item in
                       range(len(series_volume))])
    series_average_n_oris = np.array([np.mean(series_n_oris[item][int(len(series_n_oris[item]) / 2):-1]) for item in
                             range(len(series_n_oris))])
    series_average_n_initiator = np.array([np.mean(series_n_initiator[item][int(len(series_n_initiator[item]) / 2):-1]) for item in
                             range(len(series_n_initiator))])

    conc_regulator_approx = calculateApproximateRegConc(parameter_set, np.array(series_average_n_oris),
                                                        np.array(series_average_volume))
    conc_regulator_solved = np.array([fsolve(solveRegulatorConcentrationAnalytically, conc_regulator_approx[item],
                                             args=(parameter_set, series_average_n_oris[item], series_average_volume[item], item))[
                                 0] for item in range(conc_regulator_approx.size)])
    if parameter_set.version_of_titration == 'regulator_and_init_constit_expressed':
        conc_regulator_solved = parameter_set.basal_rate_regulator * series_average_n_oris / series_average_volume / parameter_set.rate_growth
    conc_initiator_approx = calculateApproxInitiatorConcIfHillCoeffsEqual(parameter_set, conc_regulator_approx)
    conc_initiator_solved = calculateApproximateInitiatorConc(parameter_set, conc_regulator_solved,
                                                              np.array(series_average_n_oris),
                                                              np.array(series_average_volume))
    v_init_approx = (parameter_set.n_c_max_0 + parameter_set.critical_free_conc) / conc_initiator_approx
    if parameter_set.version_of_model == "titration":
        v_init_th = (parameter_set.n_c_max_0 + parameter_set.critical_free_conc * series_v_init_per_ori) / conc_initiator_solved
    else:
        v_init_th = calculate_v_initi_no_overlap_switch(parameter_set)

    # plottingTools.plot_series_of_one_array(parameters.file_path,
    #                                        [x_axis], [series_average_n_oris/series_average_volume], r'$1/\tau_d$',
    #                                        r'$\rm [g]$', 'gene_density',
    #                                        labels=[r'$\rm [g]$'])
    # plottingTools.plot_series_of_one_array(parameters.file_path,
    #                                        [x_axis, x_axis], [series_average_n_oris, series_average_volume], r'$1/\tau_d$',
    #                                        r'$\rm n_{ori}, V$', 'gene_numbers_and_volume',
    #                                        labels=[r'$\rm n_{ori}$', r'$V$'])
    plottingTools.plot_series_of_one_array(parameter_set.file_path,
                                           [x_axis], [series_v_init_per_ori], r'$1/\tau_d$',
                                           r'$V^\ast/n_{\rm ori}$', 'initiation_volume_average',
                                           labels=[r'$V^\ast/n_{\rm ori}$'])
    plottingTools.plot_series_of_one_array(parameter_set.file_path,
                                           [x_axis], [series_average_n_initiator/series_average_volume], r'$1/\tau_d$',
                                           r'$[p]$', 'average_concentration_init',
                                           labels=[r'$[\bar{p}]$'])

def return_average_conc_and_growth_rates_titration_theory(parameters):
    x_axis = parameters.doubling_rate

    series_n_oris = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'n_ori')
    series_n_initiator = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'N_init')
    series_volume = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'volume')

    series_average_volume = np.array([np.mean(series_volume[item][int(len(series_volume[item]) / 2):-1]) for item in
                       range(len(series_volume))])
    series_average_n_oris = np.array([np.mean(series_n_oris[item][int(len(series_n_oris[item]) / 2):-1]) for item in
                             range(len(series_n_oris))])
    series_average_n_initiator = np.array([np.mean(series_n_initiator[item][int(len(series_n_initiator[item]) / 2):-1]) for item in
                             range(len(series_n_initiator))])

    conc_regulator_approx = calculateApproximateRegConc(parameters, np.array(series_average_n_oris),
                                                                     np.array(series_average_volume))
    conc_regulator_solved = np.array([fsolve(solveRegulatorConcentrationAnalytically, conc_regulator_approx[item],
                                    args=(parameters, series_average_n_oris[item], series_average_volume[item], item))[
                                 0] for item in range(conc_regulator_approx.size)])
    conc_initiator_approx = calculateApproxInitiatorConcIfHillCoeffsEqual(parameters, conc_regulator_approx)
    conc_initiator_solved = calculateApproximateInitiatorConc(parameters, conc_regulator_solved,
                                                                           np.array(series_average_n_oris),
                                                                           np.array(series_average_volume))
    return x_axis, series_average_n_initiator/series_average_volume

def plot_rates_together(parameters):
    concentrations = np.arange(0, parameters.michaelis_const_initiator[0] * 2, 0.01)
    print(concentrations)
    series_dil_rates = [concentrations * parameters.rate_growth[item] for item in range(parameters.n_series)]
    series_conc = [concentrations for item in range(parameters.n_series)]
    labels = [r'$\lambda = $' + str(np.round(parameters.rate_growth[item], 2)) for item in range(parameters.n_series)]
    if parameters.gene_expression_model =="standard":
        print(parameters.michaelis_const_initiator[0] * 2)
        series_n_oris = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'n_ori')
        series_volume = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'volume')
        series_average_n_oris = np.array([np.mean(series_n_oris[item][int(len(series_n_oris[item]) / 2):-1]) for item in
                                 range(len(series_n_oris))])
        series_average_volume = np.array([np.mean(series_volume[item][int(len(series_volume[item]) / 2):-1]) for item in
                           range(len(series_volume))])
        series_prod_rates = [parameters.basal_rate_regulator[item] * (series_average_n_oris[item] / series_average_volume[item]) / (1 + (concentrations / parameters.michaelis_const_regulator[item]) ** parameters.hill_coeff_regulator[item]) for item in range(parameters.n_series)]
        print(series_prod_rates)
    else:
        series_prod_rates = [
            parameters.basal_rate_initiator[item] / (
                        1 + (concentrations / parameters.michaelis_const_initiator[item]) **
                        parameters.hill_coeff_initiator[item]) for item in range(parameters.n_series)]
        print(series_prod_rates)
    plottingTools.plot_series_of_two_arrays(parameters.file_path, series_conc, series_dil_rates, series_prod_rates,
                                            r'[p]', 'rates',
                                            'rate_comparison', labels=labels)

def calculate_charac_timescale_dimless(parameters, fixed_point, delta_0):
    return 1 / (- parameters.hill_coeff_initiator * fixed_point ** (parameters.hill_coeff_initiator - 1) / (
            1 + fixed_point ** parameters.hill_coeff_initiator) ** 2 - delta_0)


def plot_charac_timescale_as_function_of_hill_coeff(parameters):
    series_n_oris = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'n_ori')
    series_volume = return_list_of_time_arrays_for_nseries(parameters, 'dataset_time_traces', 'volume')
    series_average_volume = np.array([np.mean(series_volume[item][int(len(series_volume[item]) / 2):-1]) for item in
                       range(len(series_volume))])
    series_average_n_oris = np.array([np.mean(series_n_oris[item][int(len(series_n_oris[item]) / 2):-1]) for item in
                             range(len(series_n_oris))])
    conc_regulator_approx = calculateApproximateRegConc(parameters, np.array(series_average_n_oris),
                                                                     np.array(series_average_volume))
    conc_regulator_solved = np.array([fsolve(solveRegulatorConcentrationAnalytically, conc_regulator_approx[item],
                                    args=(parameters, series_average_n_oris[item], series_average_volume[item], item))[
                                 0] for item in range(conc_regulator_approx.size)])
    delta_0 = parameters.michaelis_const_regulator / (parameters.basal_rate_regulator * series_average_n_oris / series_average_volume)
    charac_time = calculate_charac_timescale_dimless(parameters, conc_regulator_solved, delta_0)
    real_charac_time = - parameters.michaelis_const_regulator / (parameters.basal_rate_regulator * series_average_n_oris / series_average_volume) * charac_time
    plottingTools.plot_series_of_one_array(parameters.file_path,
                                           [parameters.hill_coeff_regulator], [real_charac_time], r'Hill coefficient n',
                                           r'$\tau$', 'charact_timescale',
                                           labels=[r'n'])

def calculate_average_active_fraction(data_frame, indx_cycle, indx_series):
    time_traces_data_frame = pd.read_hdf(data_frame['path_dataset'].iloc[indx_series], key='dataset_time_traces')
    time = np.array(time_traces_data_frame["time"])
    active_fraction = np.array(time_traces_data_frame["active_fraction"])
    v_d_data_frame = pd.read_hdf(data_frame['path_dataset'].iloc[indx_series], key='dataset_div_events')
    t_start = v_d_data_frame['t_d'][indx_cycle]
    print('t_start:', t_start)
    print(1/data_frame["doubling_rate"].iloc[indx_series])
    t_end =v_d_data_frame['t_d'][indx_cycle+1]
    print('t_end:', t_end)
    indx_start = np.where(time == t_start)[0][0]
    indx_end = np.where(time == t_end)[0][0]
    print('indices:', indx_start, indx_end)
    return np.average(active_fraction[indx_start:indx_end])
