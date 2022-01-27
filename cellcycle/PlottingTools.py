import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

def plot_data_frame(x, y, data_frame, file_paths, plot_label, format= 'pdf'):
    sns.lineplot(x=x, y=y, data=data_frame)
    plt.savefig(file_paths + '/' + plot_label + '.' + format, format=format)


def plot_two_arrays(file_paths, xaxis, yaxis, x_axis_label, y_axis_label, plot_label, format= 'pdf',  cv=0, theory=[], label_th=[]):
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if cv==0:
        plt.plot(xaxis, yaxis)
    else:
        mean = np.mean(yaxis)
        cv = np.std(yaxis)/mean
        plt.plot(xaxis, yaxis,  label=r'$\mu={}, CV={}$'.format(np.round(mean, 3), np.round(cv, 3)))
        for item in range(len(theory)):
            plt.plot(xaxis, theory[item] * np.ones(xaxis.size), label=label_th[item])
        plt.legend()

    plt.savefig(file_paths + '/' + plot_label + '.' + format, format=format)
    plt.clf()


def plot_four_arrays_loglog(file_paths, xaxis, yaxis, xaxis2, yaxis2, x_axis_label, y_axis_label, plot_label):
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.loglog(xaxis, yaxis)
    plt.loglog(xaxis2, yaxis2)
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def plot_four_arrays_labels(file_paths, xaxis, yaxis, label1, xaxis2, yaxis2, label2, x_axis_label, y_axis_label, plot_label):
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.plot(xaxis, yaxis, label=label1)
    plt.plot(xaxis2, yaxis2, label=label2)
    plt.legend()
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def plot_series_of_two_arrays(file_paths, xaxis, yaxis1, yaxis2, x_axis_label, y_axis_label, plot_label, labels=0, color=0, vlines=0, legend_vlines=0):
    cmap = plt.get_cmap('viridis')
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    for i_series in range(len(xaxis)):
        print(i_series)
        if color==0:
            series_color = cmap(i_series / len(xaxis))
        else:
            series_color = color[i_series]
        if labels==0:
            plt.plot(xaxis[i_series], yaxis1[i_series], color=series_color)
            plt.plot(xaxis[i_series], yaxis2[i_series], color=series_color)
        else:
            plt.plot(xaxis[i_series], yaxis1[i_series], label=labels[i_series], color=series_color)
            plt.plot(xaxis[i_series], yaxis2[i_series], color=series_color)
    if labels != 0:
        plt.legend()
    if vlines != 0:
        plt.axvline(vlines, color='r', label=legend_vlines)
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def plot_series_of_one_array(file_paths, xaxis, yaxis1, x_axis_label, y_axis_label, plot_label, labels=[], color=[]):
    cmap = plt.get_cmap('viridis')
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    for i_series in range(len(xaxis)):
        print(i_series)
        if len(color)==0:
            series_color = cmap(i_series / len(xaxis))
        else:
            series_color = color[i_series]
        if len(labels)==0:
            plt.plot(xaxis[i_series], yaxis1[i_series], color=series_color)
        else:
            plt.plot(xaxis[i_series], yaxis1[i_series], label=labels[i_series], color=series_color)
    if labels != 0:
        plt.legend()
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def plot_series_of_two_arrays_with_mean(file_paths, xaxis, yaxis, x_axis_label, y_axis_label, plot_label, parameter_set):
    plt.figure()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    times_for_plot, N_for_plot_all_series_np = make_equidistant_time_data_array(50, xaxis, yaxis)
    print('numpy array of all series: ', N_for_plot_all_series_np)
    mean_series = np.mean(N_for_plot_all_series_np, axis=1)
    mean_th = parameter_set.iloc[0]['n_0'] * np.exp(times_for_plot * parameter_set.iloc[0]['rate_growth'])
    mean_th_plot = mean_th[mean_th < parameter_set.iloc[0]['n_c_max']]
    time_th_plot = times_for_plot[mean_th < parameter_set.iloc[0]['n_c_max']]
    for i_series in range(len(xaxis)):
        plt.plot(xaxis[i_series], yaxis[i_series], color='steelblue', alpha=0.3)
    plt.plot(times_for_plot, mean_series, label=r'$\langle N \rangle$', c='r')
    plt.plot(time_th_plot, mean_th_plot, label=r'$\langle N_{th} \rangle$', c='green')
    plt.legend()
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def plot_three_subplots(file_paths, xaxis, yaxis1, yaxis2, yaxis3, x_axis_label, y_axis_label1, y_axis_label2, y_axis_label3, plot_label):

    plt.figure()
    plt.subplot(311)

    plt.ylabel(y_axis_label1)
    plt.plot(xaxis[0], yaxis1[0])

    plt.subplot(312)
    plt.ylabel(y_axis_label2)
    plt.plot(xaxis[0], yaxis2[0], 'orange')

    plt.subplot(313)
    plt.plot(xaxis[0], yaxis3[0], 'darkcyan')

    plt.ylabel(y_axis_label3)
    plt.xlabel(x_axis_label)
    plt.savefig(file_paths + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def three_subplots_n_series(file_path, xaxis_list, n_dnaa_list, v_list, density_list, x_axis_label, label_list,
                            plot_label, title, colors=[], legend1=[], legend2=[], legend3=[], vlines1=[], color_vlines1='r', vlines2=[], color_vlines2='b', hlines3=None):
    cmap = plt.get_cmap('viridis')
    plt.figure()
    plt.xlabel(x_axis_label)

    plt.subplot(311)
    plt.title(title, loc='center')
    plt.ylabel(label_list[0])
    for i_series in range(len(n_dnaa_list)):
        if len(colors)==0:
            plt.plot(xaxis_list[i_series], n_dnaa_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
        else:
            plt.plot(xaxis_list[i_series], n_dnaa_list[i_series], color=colors[i_series])
    print('length legend', len(legend1))
    if len(legend1) > 0:
        for i_series in range(len(vlines1)):
            if i_series==0:
                plt.axvline(vlines1[i_series], color=color_vlines1, label=legend1[0])
            else:
                plt.axvline(vlines1[i_series], color=color_vlines1)
        for i_series in range(len(vlines2)):
            if i_series == 0:
                plt.axvline(vlines2[i_series], color=color_vlines2, label=legend1[1])
            else:
                plt.axvline(vlines2[i_series], color=color_vlines2)
        plt.legend()
    plt.subplot(312)
    plt.ylabel(label_list[1])
    for i_series in range(len(v_list)):
        if len(legend2) == 0:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
            else:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=colors[i_series])
        else:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)), label=legend2[i_series])
            else:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=colors[i_series],
                         label=legend2[i_series])
            plt.legend()
    for i_series in range(len(vlines1)):
        plt.axvline(vlines1[i_series], color=color_vlines1)
    for i_series in range(len(vlines2)):
        plt.axvline(vlines2[i_series], color=color_vlines2)
    plt.subplot(313)
    plt.ylabel(label_list[2])
    for i_series in range(len(density_list)):
        if len(legend3) == 0:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
            else:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=colors[i_series])
        else:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)), label=legend3[i_series])
            else:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=colors[i_series],
                         label=legend3[i_series])
            plt.legend()
    for i_series in range(len(vlines1)):
        plt.axvline(vlines1[i_series], color=color_vlines1)
    for i_series in range(len(vlines2)):
        plt.axvline(vlines2[i_series], color=color_vlines2)
    if hlines3 is not None:
        plt.plot(xaxis_list[0], hlines3 * np.ones(xaxis_list[0].size,))
    plt.savefig(file_path + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def four_subplots_aligned_n_series(file_path, xaxis_list, n_dnaa_list, v_list, free_list, density_list, x_axis_label,
                                   label_list, plot_label, title, colors=[], legend1=[], legend2=[], legend3=[], legend4=[],
                                   vlines1=[], color_vlines1='r', vlines2=[], color_vlines2='b'):
    cmap = plt.get_cmap('viridis')
    plt.figure()
    plt.xlabel(x_axis_label)

    plt.subplot(411)
    plt.title(title, loc='center')
    plt.ylabel(label_list[0])
    for i_series in range(len(n_dnaa_list)):
        if len(colors)==0:
            plt.plot(xaxis_list[i_series], n_dnaa_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
        else:
            plt.plot(xaxis_list[i_series], n_dnaa_list[i_series], color=colors[i_series])
    print('length legend', len(legend1))
    if len(legend1) > 0:
        for i_series in range(len(vlines1)):
            if i_series==0:
                plt.axvline(vlines1[i_series], color=color_vlines1, label=legend1[0])
            else:
                plt.axvline(vlines1[i_series], color=color_vlines1)
        for i_series in range(len(vlines2)):
            if i_series == 0:
                plt.axvline(vlines2[i_series], color=color_vlines2, label=legend1[1])
            else:
                plt.axvline(vlines2[i_series], color=color_vlines2)
        plt.legend()

    plt.subplot(412)
    plt.ylabel(label_list[1])
    for i_series in range(len(v_list)):
        if len(legend2) == 0:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
            else:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=colors[i_series])
        else:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)), label=legend2[i_series])
            else:
                plt.plot(xaxis_list[i_series], v_list[i_series], color=colors[i_series], label=legend2[i_series])
            plt.legend()
    for i_series in range(len(vlines1)):
        plt.axvline(vlines1[i_series], color=color_vlines1)
    for i_series in range(len(vlines2)):
        plt.axvline(vlines2[i_series], color=color_vlines2)

    plt.subplot(413)
    plt.ylabel(label_list[2])
    for i_series in range(len(free_list)):
        if len(legend3) == 0:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], free_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
            else:
                plt.plot(xaxis_list[i_series], free_list[i_series], color=colors[i_series])
        else:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], free_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)), label=legend3[i_series])
            else:
                plt.plot(xaxis_list[i_series], free_list[i_series], color=colors[i_series],
                         label=legend3[i_series])
            plt.legend(loc='upper left')
    for i_series in range(len(vlines1)):
        plt.axvline(vlines1[i_series], color=color_vlines1)
    for i_series in range(len(vlines2)):
        plt.axvline(vlines2[i_series], color=color_vlines2)
    # plt.plot(0, 0)

    plt.subplot(414)
    plt.ylabel(label_list[3])
    for i_series in range(len(density_list)):
        if len(legend4) == 0:
            if len(colors) == 0:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)))
            else:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=colors[i_series])
        else:
            if len(colors)==0:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / (len(n_dnaa_list)+1)), label=legend4[i_series])
            else:
                plt.plot(xaxis_list[i_series], density_list[i_series], color=colors[i_series],
                         label=legend4[i_series])
            plt.legend(loc='upper left')
    for i_series in range(len(vlines1)):
        plt.axvline(vlines1[i_series], color=color_vlines1)
    for i_series in range(len(vlines2)):
        plt.axvline(vlines2[i_series], color=color_vlines2)
    plt.savefig(file_path + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()


def two_subplots_n_series(file_path, xaxis_list, n_dnaa_list, v_list, density_list, x_axis_label, label_list, plot_label, title, parameter_list, hlines_a=[], hlines_p_0=[], hlines_phi_0=[], hlines_p=[], hlines_phi=[]):
    cmap = plt.get_cmap('viridis')
    time = np.array(xaxis_list[0])
    tmax = time[-1]
    print('tmax ', tmax)
    plt.figure()
    plt.xlabel(x_axis_label)

    plt.subplot(211)
    plt.ylabel(label_list[1])
    for i_series in range(len(v_list)):
        plt.plot(xaxis_list[i_series], v_list[i_series], color=cmap(i_series / len(n_dnaa_list)))
        if hlines_p !=[]:
            # if i_series==0:
            plt.hlines(hlines_p_0[i_series], 0, tmax, linestyle='--', color=cmap(i_series / len(n_dnaa_list)), label=r'$p^{0}$=' + str(np.round(hlines_p_0[i_series], 2)))
            plt.hlines(hlines_p[i_series], 0, tmax, linestyle='-', color=cmap(i_series / len(n_dnaa_list)), label=r'$p^{eq}$=' + str(np.round(hlines_p[i_series], 2)))
            plt.legend()
    plt.subplot(212)
    plt.ylabel(label_list[2])
    for i_series in range(len(density_list)):
        density = np.array(density_list[i_series])
        print('density ', density)
        print('size ', density.size)
        print('half array ', density[int(density.size/2):])
        density = np.mean(density[int(density.size/2):])
        # density = density[-1]
        # plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / len(n_dnaa_list)))
        plt.plot(xaxis_list[i_series], density_list[i_series], color=cmap(i_series / len(n_dnaa_list)), label=r'$\bar{N}_D=$ ' + str(np.round(density,2)))
        plt.legend()
        if hlines_phi !=[]:
            if i_series==0:
                plt.hlines(hlines_phi_0[i_series], 0, tmax, linestyle='--', color=cmap(i_series / len(n_dnaa_list)), label=r'$\phi_R^{0}=$' + str(np.round(hlines_phi_0[i_series], 2)))
                plt.hlines(hlines_phi[i_series], 0, tmax, linestyle='-', color=cmap(i_series / len(n_dnaa_list)), label=r'$\phi_R^{eq}=$' + str(np.round(hlines_phi[i_series], 2)))
            plt.legend()

    plt.savefig(file_path + '/' + plot_label + '.pdf', format='pdf')
    plt.clf()

def four_subplots_n_series(file_path, xaxis_list, v_list1, density_list1,v_list2, density_list2, x_axis_label, label_list, plot_label, parametertitle, parameterlist):
    cmap = plt.get_cmap('viridis')
    time = np.array(xaxis_list[0])
    tmax = time[-1]
    print('tmax ', tmax)
    plt.figure()
    plt.xlabel(x_axis_label)

    plt.subplot(2, 2, 1)
    plt.ylabel(label_list[1])
    for i_series in range(len(v_list1)):
        plt.plot(xaxis_list[i_series], v_list1[i_series], color=cmap(i_series / len(v_list1)))

    plt.subplot(2, 2, 2)
    plt.ylabel(label_list[2])
    for i_series in range(len(v_list1)):
        density = np.array(density_list1[i_series])
        density = np.mean(density[int(density.size/2):])
        plt.plot(xaxis_list[i_series], density_list1[i_series], color=cmap(i_series / len(v_list1)), label=r'$\bar{N}_D=$ ' + str(np.round(density,0)))
        plt.legend()

    plt.subplot(2, 2, 3)
    plt.ylabel(label_list[1])
    for i_series in range(len(v_list2)):
        plt.plot(xaxis_list[i_series], v_list2[i_series], color=cmap(i_series / len(v_list1)))

    plt.subplot(2, 2, 4)
    plt.ylabel(label_list[2])
    for i_series in range(len(density_list2)):
        density = np.array(density_list2[i_series])
        density = np.mean(density[int(density.size/2):])
        plt.plot(xaxis_list[i_series], density_list2[i_series], color=cmap(i_series / len(v_list1)), label=r'$1/\tau_d=$ '+ str(parameterlist[i_series]) + r'$, \bar{N}_D=$ ' + str(np.round(density,0)))
        plt.legend()

    plt.savefig(file_path + '/' + plot_label + '.svg', format='svg')
    plt.clf()


def plot_CV_for_all_N_max(file_paths, time_array, data_array, x_axis_label, y_axis_label, plot_label, parameter_set):
    std = np.std(time_array, axis=0)
    mean = np.mean(time_array, axis=0)
    mean = mean[1:]
    std = std[1:]
    cv = std/mean
    data = data_array[0]
    data_plot = data[1:]
    print('standard dev', std)
    print('mean', mean)
    print('cv and n ', cv, data_plot)
    plot_two_arrays(file_paths, data_plot, cv, x_axis_label, y_axis_label, plot_label)
    plt.clf()


def make_equidistant_time_data_array(N_tot, time_array, data_array):
    flat_list_time = [item for sublist in time_array for item in sublist]
    t_max = np.amax(flat_list_time)
    dt = t_max / N_tot
    times_for_plot = np.arange(0, t_max, dt)
    N_for_plot_all_series = []
    print('times for plot: ', times_for_plot)
    for time_i in range(0, times_for_plot.size):
        N_for_plot = []
        print('we are at time ', times_for_plot[time_i], ' and want to know N of all series at that time')
        for item in range(0, len(time_array)):
            print('series number ', item)
            time_array_np_i = np.array(time_array[item])
            data_array_np_i = np.array(data_array[item])
            print('time and N array of this series: ', time_array_np_i, data_array_np_i)
            print('check for times smaller or equal to ', times_for_plot[time_i])
            N_at_time_i = data_array_np_i[time_array_np_i <= times_for_plot[time_i]]
            times_at_time_i = time_array_np_i[time_array_np_i <= times_for_plot[time_i]]
            print('times and N smaller than this time: ', times_at_time_i, N_at_time_i)
            print('last values of N ', N_at_time_i[-1])
            N_for_plot.append(N_at_time_i[-1])
        print('N for plot: ', N_for_plot)
        N_for_plot_all_series.append(N_for_plot)
    print('N for plot all series: ', N_for_plot_all_series)
    # take mean and std from list of arrays for each time_for_plot index
    N_for_plot_all_series_np = np.array(N_for_plot_all_series)
    return times_for_plot, N_for_plot_all_series_np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def set_value_to_array(index_init, index_end, index_division, value_array):
    t_init = index_division[index_init]
    t_end = index_division[index_end]
    size = (t_end - t_init)
    value_array = np.ones(size) * value_array[index_init]
    return value_array


def division_time_mask(t_div_arr, time_arr, digit):
    t_div_arr_round = np.round(t_div_arr, digit)
    masked_division = np.in1d(time_arr, t_div_arr_round)  # True, if element of time_arr is equal to division time
    index_division = np.argwhere(masked_division == True)
    return masked_division, index_division