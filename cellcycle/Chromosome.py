import numpy as np


class Chromosome:
    def __init__(self, parameter_dict, t_init, v_init, t_end_blocked=0, t_end_blocked_production=0, active_rep=1, length=0, sites=0, blocked=True, blocked_production=True, rida=2):
        self.parameter_dict = parameter_dict
        # changing chromosome variables
        self.t_init = t_init
        self.v_init = v_init
        self.division_time_RDA = t_init + self.calculate_next_division_time_RDA(self.v_init)
        # print('initiated, division time set:', self.division_time_RDA, ' current volume:', self.v_init)
        self.t_end_blocked = t_end_blocked
        self.t_end_blocked_production = t_end_blocked_production
        self.time = np.array([self.t_init])
        self.length = np.array([length])
        if self.parameter_dict.homogeneous_dist_sites == 0:
            self.sites = np.array([self.parameter_dict.n_c_max_0])  # if distribution is inhomogeneous, all sites are created instantaneously after the origin has been replicated
        else:
            self.sites = np.array([sites])  # if distribution is homogeneous, the number of sites depends on how the chromosome is initiated, either fully replicated or being actively replicated
        self.active_replication = active_rep
        self.division_time = self.parameter_dict.t_max

        # set origin blocked or unblocked for initiating new round of replication
        if self.parameter_dict.period_blocked > 0:
            self.blocked = blocked
        else:
            self.blocked = False

        # set DnaA promoter blocked or unblocked for initiating new round of replication
        if self.parameter_dict.period_blocked_production > 0:
            self.blocked_production = blocked_production
        else:
            self.blocked_production = False

        # changing ultrasensitivity parameters
        self.n_rida = rida
        if self.active_replication ==0:
            self.n_dars2 = 1
            self.n_dars1 = 1
            self.n_datA = 1
        else:
            self.n_dars2 = 0
            self.n_dars1 = 0
            self.n_datA = 0
        self.n_dars2_high_activity = 0
        self.n_datA_high_activity = 0

    def replicate(self, time):
        if self.active_replication == 1:
            if self.length[-1] < 1:
                # print(' length before was ', self.length[-1])
                new_length = self.length[-1] + self.parameter_dict.rate_rep * self.parameter_dict.time_step
                self.length = np.append(self.length, new_length)
                if self.parameter_dict.homogeneous_dist_sites == 1:
                    new_sites = self.sites[-1] + self.parameter_dict.rate_synth_sites * self.parameter_dict.time_step
                    self.sites = np.append(self.sites, new_sites)
                    # print('replicated new sites',  self.rate_synth_sites * self.dt)
                else:
                    self.sites = np.append(self.sites, self.parameter_dict.n_c_max_0)
                self.time = np.append(self.time, time)
                # print('new sites', self.rate_synth_sites* self.dt)
                # print(' length after replication was ', self.length[-1])
                if self.length[-1] > 1 or self.sites[-1] > self.parameter_dict.n_c_max_0:
                    self.length[-1] = 1
                    self.sites[-1] = self.parameter_dict.n_c_max_0
                # Check whether dars1, dars2 and datA are being doubled
                if self.length[-1] >= self.parameter_dict.site_dars2: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_dars2 = 1
                if self.length[-1] >= self.parameter_dict.site_dars1: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_dars1 = 1
                if self.length[-1] >= self.parameter_dict.site_datA: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    self.n_datA = 1
                # Check whether datA and dars2 are in high or low activity state
                if self.length[-1] >= self.parameter_dict.relative_chromosome_position_onset_dars2: #as soon as we reach position where dars2 becomes more active, the number of high activity dars2 goes up to two
                    self.n_dars2_high_activity = 2
                if self.length[-1] >= self.parameter_dict.relative_chromosome_position_onset_datA: #as soon as we reach position where datA becomes more active, the number of high activity datA goes up to two
                    self.n_datA_high_activity = 2
                if self.length[-1] >= self.parameter_dict.relative_chromosome_position_offset_dars2: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    # print('dars was doubled')
                    self.n_dars2_high_activity = 0
                if self.length[-1] >= self.parameter_dict.relative_chromosome_position_offset_datA: #as soon as we reach position of site of dars2, the number of dars2 sites goes from 0 to 1
                    # print('dars was doubled')
                    self.n_datA_high_activity = 0

            else:
                # replication is not active anymore and therefore rida is switched off
                self.active_replication = 0
                self.n_rida = 0
                if self.parameter_dict.version_of_coupled_division_regulation == 'RDA':
                    # print('replication finished, set now division time')
                    # print('Division time RDA: ',self.division_time_RDA, 'time at the moment:', time)
                    if time < self.division_time_RDA:
                        self.division_time = self.division_time_RDA
                    else:
                        self.division_time = time
                        # print('divide now')
                else:
                    self.division_time = time + self.parameter_dict.t_D
                # print('replication was finished, division time was set')

    #  checks whether origin blocked, if condition correct unblocks, returns 0 if not and 1 if it is blocked
    def check_blocked(self, time):
        if time >= self.t_end_blocked:
            self.blocked = False
            # print('unblocked')
            return self.blocked
        else:
            self.blocked = True
            return self.blocked

    #  checks whether DnaA production is blocked, if condition correct unblocks, returns 0 if not and 1 if it is blocked
    def check_blocked_production(self, time):
        if time >= self.t_end_blocked_production:
            self.blocked_production = False
            # print('unblocked')
            return self.blocked_production
        else:
            self.blocked_production = True
            return self.blocked_production
    def store_everything(self):
        return self.time, self.length, self.sites

    def set_t_end_blocked(self, t_end_blocked):
        self.t_end_blocked = t_end_blocked

    def set_t_end_blocked_production(self, t_end_blocked_production):
        self.t_end_blocked_production = t_end_blocked_production

    def calculate_next_division_time_RDA(self, volume):
        average_added_volume = self.parameter_dict.v_init_th * (np.exp(self.parameter_dict.rate_growth * self.parameter_dict.t_CD)-1)
        noisy_added_volume = average_added_volume + np.random.normal(0, self.parameter_dict.cv_added_volume) * average_added_volume
        noisy_division_volume = noisy_added_volume + volume
        return np.log(noisy_division_volume/volume)/self.parameter_dict.rate_growth