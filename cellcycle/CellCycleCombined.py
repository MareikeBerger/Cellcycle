import numpy as np
from treelib import Node, Tree
import pandas as pd

from . import PlottingTools as plottingTools
from .Chromosome import Chromosome


class CellCycleCombined:
    def __init__(self, parameter_dict):
        """ This class is used for running several cell cycles with the given parameters obtained by the dictionary parameter_dict.

        Parameters
        ----------
        parameter_dict : dictionary
            The parameter dictionary is used during this entire class to obtain all relevant cell cycle parameters
        """
        self.parameter_dict = parameter_dict

        # changing variables from ultrasensitivity model
        self.min_frac = self.parameter_dict.init_conc / self.parameter_dict.total_conc

        # changing variables of one cell line, time goes until t_max
        self.time = np.arange(0, self.parameter_dict.t_max, self.parameter_dict.time_step)
        self.volume = np.ones(self.time.size) * self.parameter_dict.v_0
        self.N_init = np.ones(self.time.size) * self.parameter_dict.n_init_0
        self.N_regulator = np.ones(self.time.size) * self.parameter_dict.n_regulator_0
        self.N_free = np.ones(self.time.size)
        self.free_conc = self.N_free / self.volume
        self.total_conc = self.N_init / self.volume
        self.n_ori = np.ones(self.time.size) * self.parameter_dict.n_ori_0
        self.sites_total = np.ones(self.time.size) * self.parameter_dict.n_c_max_0
        self.calculate_free_dnaA_concentration(0)
        self.length_total = np.ones(self.time.size)
        self.N_c_max = self.parameter_dict.n_c_max_0
        self.next_div_t = self.parameter_dict.t_max
        self.next_stoch_prod_time = self.parameter_dict.t_max
        self.active_fraction = np.zeros(self.time.size)
        self.active_conc = self.active_fraction * self.parameter_dict.total_conc
        self.print_verbose('active fraction: {}'.format(self.active_fraction))
        self.N_active = np.zeros(self.time.size)
        self.activation_rate_lipids_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_lipids
        self.deactivation_rate_datA_tot = np.ones(self.time.size) * self.parameter_dict.deactivation_rate_datA
        self.activation_rate_dars2_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_dars2
        self.activation_rate_dars1_tot = np.ones(self.time.size) * self.parameter_dict.activation_rate_dars1
        self.deactivation_rate_rida_tot = np.ones(self.time.size) * self.parameter_dict.deactivation_rate_rida
        self.t_end_blocked = -1
        self.chromosome_dict = {"chromosome 0": Chromosome(self.parameter_dict, t_init=0, v_init=self.parameter_dict.v_init_th, active_rep=0, length=1,
                                                           sites=self.parameter_dict.n_c_max_0, blocked=0, blocked_production=0, rida=0)}
        self.chromosome_tree = Tree()
        self.chromosome_tree.create_node("Chromosome 0", "chromosome 0")  # root node

        # variables related to explicitly modelling lipids
        self.N_lipids = np.ones(self.time.size) * self.parameter_dict.activation_rate_lipids * self.parameter_dict.v_0
        self.N_regulator_lipids = np.ones(self.time.size) * self.parameter_dict.activation_rate_lipids * self.parameter_dict.v_0
        self.lipid_conc = np.ones(self.time.size) * self.parameter_dict.lipid_conc_0 # * self.parameter_dict.activation_rate_lipids/self.parameter_dict.rate_growth
        self.one_time_perturbation = 0

        # variables to model noise in RIDA
        self.rida_deactivation_rate = np.ones(self.time.size) * self.parameter_dict.deactivation_rate_rida
        self.rida_noise = np.zeros(self.time.size)

        # variables required in the case of external oscillations of DnaA
        self.N_init_ext = self.parameter_dict.amplitude_oscillations * np.ones(self.time.size)

        # stored chromosomes after division events
        self.stored_sites = []
        self.stored_lengths = []
        self.stored_times = []

        # variable for aborting simulations
        self.abort_simulation = False

        # store initiation times, concentrations, numbers of dnaa
        self.t_init = []
        self.t_div = []
        self.n_init = []
        self.v_init = []
        self.n_sites_init = []
        self.n_ori_init = []
        self.v_init_per_ori = []
        self.c_free_init = []
        self.frac_active_init = []
        self.lipid_conc_init = []
        self.n_active_init = []
        self.v_b = np.array([])
        self.v_b_before_init = []
        self.last_v_b = None
        self.list_of_tuples_v_b_v_d = []
        self.next_division_volume = self.parameter_dict.division_volume

        self.data_frame_cellcycle = self.makeDataFrameOfCellCycle()

    def makeDataFrameOfCellCycle(self):
        return pd.DataFrame({"min_frac": self.min_frac,
                             "time": self.time,
                             "volume": self.volume,
                             "N_init": self.N_init,
                             "N_regulator": self.N_regulator,
                             "N_lipids": self.N_lipids,
                             "lipid_conc": self.lipid_conc,
                             "N_regulator_lipids": self.N_regulator_lipids,
                             "N_free": self.N_free,
                             "free_conc": self.free_conc,
                             "n_ori": self.n_ori,
                             "sites_total": self.sites_total,
                             "length_total": self.length_total,
                             "active_conc": self.active_conc,
                             "active_fraction": self.active_fraction,
                             "activation_rate_lipids_tot": self.activation_rate_lipids_tot,
                             "deactivation_rate_datA_tot": self.deactivation_rate_datA_tot,
                             "activation_rate_dars2_tot": self.activation_rate_dars2_tot,
                             "activation_rate_dars1_tot": self.activation_rate_dars1_tot,
                             "deactivation_rate_rida_tot": self.deactivation_rate_rida_tot,
                             "abort_simulation": self.abort_simulation
                             })

    def makeDataFrameOfInitEvents(self):
         df_init=pd.DataFrame({
            "t_init": self.t_init,
            "n_init": self.n_init,
            "n_sites_init": self.n_sites_init,
            "v_init": self.v_init,
            "n_ori_init": self.n_ori_init,
            "v_init_per_ori": self.v_init_per_ori,
            "c_free_init": self.c_free_init,
            "frac_active_init": self.frac_active_init,
            "n_active_init": self.n_active_init,
            "v_b_before_init": self.v_b_before_init,
            "lipid_conc_init": self.lipid_conc_init
        })
         self.print_verbose('data_frame initiation event {}'.format(df_init))
         return df_init

    def makeDataFrameOfDivisionEvents(self):
        division_data_frame = pd.DataFrame(self.list_of_tuples_v_b_v_d, columns=['v_b', 'v_d', 't_d'])
        self.print_verbose('dataframe in make division event {}'.format(division_data_frame))
        return division_data_frame

    def updateVariablesUsingDataframe(self, dataframe):
        self.min_frac = dataframe["min_frac"]
        self.time = dataframe["time"]
        self.volume = dataframe["volume"]
        self.N_init = dataframe["N_init"]
        self.N_regulator = dataframe["N_regulator"]
        self.N_lipids = dataframe["N_lipids"]
        self.N_regulator_lipids = dataframe["N_regulator_lipids"]
        self.N_free = dataframe["N_free"]
        self.free_conc = dataframe["free_conc"]
        self.n_ori = dataframe["n_ori"]
        self.sites_total = dataframe["sites_total"]
        self.length_total = dataframe["length_total"]
        self.active_conc = dataframe["active_conc"]
        self.active_fraction = dataframe["active_fraction"]
        self.activation_rate_lipids_tot = dataframe["activation_rate_lipids_tot"]
        self.deactivation_rate_datA_tot = dataframe["deactivation_rate_datA_tot"]
        self.activation_rate_dars2_tot = dataframe["activation_rate_dars2_tot"]
        self.activation_rate_dars1_tot = dataframe["activation_rate_dars1_tot"]
        self.deactivation_rate_rida_tot = dataframe["deactivation_rate_rida_tot"]
        self.t_init = dataframe["t_init"]
        self.t_div = dataframe["t_div"]
        self.n_init = dataframe["n_init"]
        self.n_sites_init = dataframe["n_sites_init"]
        self.v_init = dataframe["v_init"]
        self.n_ori_init = dataframe["n_ori_init"]
        self.v_init_per_ori = dataframe["v_init_per_ori"]
        self.c_free_init = dataframe["c_free_init"]
        self.frac_active_init = dataframe["frac_active_init"]
        self.n_active_init = dataframe["n_active_init"]
        self.v_b = dataframe["v_b"]

    def print_verbose(self, input):
        if self.parameter_dict.verbose == 1:
            print(input)

    def calculate_active_frac(self, volume, total_conc, diss_const_activation, diss_const_deactivation, activation_rate, deactivation_rate):
        a = activation_rate * volume - deactivation_rate
        b = deactivation_rate * (1 + diss_const_activation / total_conc) \
            - activation_rate * volume * (1 - diss_const_deactivation / total_conc)
        c = - activation_rate * volume * diss_const_deactivation / total_conc
        return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    def grow_cell(self, n_step, growth_rate=0):
        """ The volume of the n_step is updated according to exponential growth.

        Parameters ----------
        n_step : double
            The nth step of the simulation
        growth_rate: double
            The growth rate is either given, in this case we use the provided growth rate. If no growth rate is
            given, the growth rate from the parameter dictionary is used to update the volume of the cell
        """
        if growth_rate == 0:
            self.volume[n_step] = self.volume[n_step - 1] + self.volume[
                n_step - 1] * self.parameter_dict.rate_growth * self.parameter_dict.time_step
        else:
            self.volume[n_step] = self.volume[n_step - 1] + self.volume[
                n_step - 1] * growth_rate * self.parameter_dict.time_step

    def produce_repressed_dnaA_deterministic(self, n_step, time):
        """ The total number of initiator and regulator proteins is updated in this step. If the parameter
            stochastic_gene_expression==1, noise is added to the production rate.
            Proteins in this function are produced according to the standard model of gene expression if parameter gene_expression_model = 'standard' (the
            change in the number of proteins is proportional to the number of genes). Alternatively, if the parameter gene_expression_model = 'ribo_limiting',
            the basal rate is additionally multiplied by the volume of the cell.
            Because the production rate is in both cases proportional to the gene copy number, we need to loop
            over all existing chromosomes in the cell.

        Parameters
        ----------
        n_step : double
            The nth step of the simulation
        time : double
            The time of the current simulation step

        Returns
        -------
        noise dictionary : dictionary
            This function returns a dictionary of the amount of initiator and regulator produced and of the
            noise in this time step in the initiator and in the regulator.
        """
        dn_init = 0  # this is the total number of initiators added in this step
        dn_regulator = 0  # this is the total number of regulators added in this step
        noise_init = 0  # this is the noise in the number of initiators added in this step
        noise_reg = 0  # this is the noise in the number of regulators added in this step
        dn_init_ext = 0 # this is the change in the initiator number when DnaA is expressed via an inducer from a plasmid
        produce_proteins = 1
        gene_fraction = 1
        if self.parameter_dict.gene_expression_model == 'ribo_limiting':
            if self.parameter_dict.finite_dna_replication_rate == 1:
                self.update_length(n_step)
                gene_fraction = self.count_n_oris()/self.length_total[n_step]
            basal_rate_regulator = self.parameter_dict.basal_rate_regulator * gene_fraction * self.volume[n_step]
            basal_rate_initiator = self.parameter_dict.basal_rate_initiator * gene_fraction * self.volume[n_step]
            for node_i in self.chromosome_tree.expand_tree(mode=2):
                if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                    if self.parameter_dict.block_production == 1 and self.chromosome_dict[node_i].check_blocked_production(
                            time) == True:
                        produce_proteins = 0

            if produce_proteins == 1:
                if self.parameter_dict.stochastic_gene_expression == 1:
                    noise_init = np.random.normal(0, 1) * np.sqrt(
                        2 * self.parameter_dict.noise_strength_total_dnaA * self.parameter_dict.time_step)
                    noise_reg = np.random.normal(0, 1) * np.sqrt(
                        2 * self.parameter_dict.noise_strength_total_dnaA * self.parameter_dict.time_step)
                if self.parameter_dict.version_of_titration == 'regulator_is_not_initiator':
                    dn_init = dn_init + basal_rate_initiator / (
                            1 + ((self.N_regulator[n_step - 1] / self.volume[
                        n_step - 1]) / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                    dn_regulator = dn_regulator + basal_rate_regulator / (
                            1 + ((self.N_regulator[n_step - 1] /
                                  self.volume[
                                      n_step - 1]) / self.parameter_dict.michaelis_const_regulator) ** self.parameter_dict.hill_coeff_regulator) * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_titration == 'regulator_is_initiator':
                    dn_init = dn_init + basal_rate_initiator / (1 + (self.free_conc[
                                                                         n_step - 1] / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_titration == 'different_promoters':
                    if self.parameter_dict.cooperativity == 1:
                        dn_init = dn_init + basal_rate_initiator / (1 + (self.free_conc[
                                                                         n_step - 1] / self.parameter_dict.michaelis_const_adp_dnaa) ** self.parameter_dict.hill_coeff_adp_dnaa
                                                                    + (self.free_conc[
                                                                         n_step - 1] / self.parameter_dict.michaelis_const_adp_dnaa) ** self.parameter_dict.hill_coeff_adp_dnaa
                                                                  * (self.free_conc[
                                                                         n_step - 1] * self.active_fraction[n_step - 1] / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                    else:
                        dn_init = dn_init + basal_rate_initiator / ((1 + (self.free_conc[
                                                                             n_step - 1] / self.parameter_dict.michaelis_const_adp_dnaa) ** self.parameter_dict.hill_coeff_adp_dnaa)
                                                                    * (1+ (self.free_conc[
                                                                           n_step - 1] * self.active_fraction[n_step - 1]  / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator)) * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_titration == 'regulator_and_init_constit_expressed':
                    dn_init = dn_init + basal_rate_initiator * self.parameter_dict.time_step
                    dn_regulator = dn_regulator + basal_rate_regulator * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_titration == 'regulator_and_init_prop_to_v':
                    dn_init = dn_init + basal_rate_initiator * self.volume[n_step - 1] * self.parameter_dict.time_step
                    dn_regulator = dn_regulator + basal_rate_regulator * self.volume[
                        n_step - 1] * self.parameter_dict.time_step
                else:
                    print('neither of the three titration model versions is fulfilled, stop and check this!')
                    exit()
        # using standard model of gene expression here
        else:
            basal_rate_regulator = self.parameter_dict.basal_rate_regulator
            basal_rate_initiator = self.parameter_dict.basal_rate_initiator
            if self.parameter_dict.stochastic_gene_expression == 1:
                noise_init = np.random.normal(0,1) * np.sqrt(2 * self.parameter_dict.noise_strength_total_dnaA * self.parameter_dict.time_step)
                noise_reg = np.random.normal(0,1) * np.sqrt(2 * self.parameter_dict.noise_strength_total_dnaA * self.parameter_dict.time_step)
            for node_i in self.chromosome_tree.expand_tree(mode=2):
                if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                    if self.parameter_dict.block_production == 1 and self.chromosome_dict[node_i].check_blocked_production(
                            time) == True:
                        produce_proteins = 0
                    if produce_proteins == 1:
                        if self.parameter_dict.version_of_titration == 'regulator_is_not_initiator':
                            dn_init = dn_init + basal_rate_initiator / (
                                    1 + ((self.N_regulator[n_step - 1] / self.volume[
                                n_step - 1]) / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                            dn_regulator = dn_regulator + basal_rate_regulator / (
                                    1 + ((self.N_regulator[n_step - 1] /
                                          self.volume[
                                              n_step - 1]) / self.parameter_dict.michaelis_const_regulator) ** self.parameter_dict.hill_coeff_regulator) * self.parameter_dict.time_step
                        elif self.parameter_dict.version_of_titration == 'regulator_is_initiator':
                            dn_init = dn_init + basal_rate_initiator / (1 + (self.free_conc[n_step - 1] / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                        elif self.parameter_dict.version_of_titration == 'different_promoters':
                            if self.parameter_dict.cooperativity == 1:
                                dn_init = dn_init + basal_rate_initiator / (1 + (self.free_conc[
                                                                                     n_step - 1] / self.parameter_dict.michaelis_const_adp_dnaa) ** self.parameter_dict.hill_coeff_adp_dnaa
                                                                            * (self.free_conc[
                                                                                   n_step - 1] * self.active_fraction[n_step - 1]  / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator) * self.parameter_dict.time_step
                            else:
                                dn_init = dn_init + basal_rate_initiator / ((1 + (self.free_conc[
                                                                                      n_step - 1] / self.parameter_dict.michaelis_const_adp_dnaa) ** self.parameter_dict.hill_coeff_adp_dnaa)
                                                                            * (1 + (self.free_conc[
                                                                                        n_step - 1] * self.active_fraction[n_step - 1]  / self.parameter_dict.michaelis_const_initiator) ** self.parameter_dict.hill_coeff_initiator)) * self.parameter_dict.time_step

                        elif self.parameter_dict.version_of_titration == 'regulator_and_init_constit_expressed':
                            dn_init = dn_init + basal_rate_initiator * self.parameter_dict.time_step
                            dn_regulator = dn_regulator + basal_rate_regulator * self.parameter_dict.time_step
                        elif self.parameter_dict.version_of_titration == 'regulator_and_init_prop_to_v':
                            dn_init = dn_init + basal_rate_initiator * self.volume[n_step - 1] * self.parameter_dict.time_step
                            dn_regulator = dn_regulator + basal_rate_regulator * self.volume[n_step - 1] * self.parameter_dict.time_step
                        else:
                            print('neither of the three titration model versions is fulfilled, stop and check this!')
                            exit()
        if self.parameter_dict.external_oscillations == 1:
            if self.parameter_dict.continuous_oscillations == 1:
                dn_init_ext = self.parameter_dict.offset_oscillations * gene_fraction * self.volume[
                        n_step] * self.parameter_dict.time_step + self.parameter_dict.amplitude_oscillations * (np.cos(time * 2 * np.pi / self.parameter_dict.period_oscillations) + 1) \
                                  * self.parameter_dict.time_step
            else:
                if np.cos(time * 2 * np.pi / self.parameter_dict.period_oscillations) >= 0:
                    dn_init_ext = 2 * self.parameter_dict.amplitude_oscillations * self.parameter_dict.time_step
                else:
                    dn_init_ext = 0
            if self.parameter_dict.underexpression_oscillations == 1:
                dn_init = 0
        self.N_init_ext[n_step] = self.N_init_ext[n_step - 1] + dn_init_ext
        self.N_init[n_step] = self.N_init[n_step - 1] + dn_init + noise_init + dn_init_ext
        self.N_regulator[n_step] = self.N_regulator[n_step - 1] + dn_regulator + noise_reg
        return {"dn_init": dn_init + dn_init_ext, "noise_init": noise_init, "dn_regulator": dn_regulator, "noise_reg": noise_reg}

    def produce_linear_dnaA_deterministic(self, n_step, time):
        dn = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                if not self.chromosome_dict[node_i].check_blocked_production(time):
                    dn = dn + self.parameter_dict.rate_growth * self.parameter_dict.n_c_max_0 * self.parameter_dict.time_step
        self.N_init[n_step] = self.N_init[n_step - 1] + dn

    def produce_lipids(self, n_step):
        """ The total number of lipids per cell is evolved explicitly in this function.
            Different version of the lipid production rate are possible and depend on
            how it was specified in the parameter version_of_lipid_regulation.


        Parameters
        ----------
        n_step : double
             The nth step of the simulation
        """
        dn_lipids = 0
        dn_regulator_lipids = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                if self.parameter_dict.version_of_lipid_regulation == 'rl':
                    dn_lipids = dn_lipids + self.parameter_dict.basal_rate_lipids / (
                            1 + ((self.N_regulator_lipids[n_step - 1] / self.volume[
                        n_step - 1]) / self.parameter_dict.michaelis_const_lipids) ** self.parameter_dict.hill_coeff_lipids) * self.parameter_dict.time_step
                    dn_regulator_lipids = dn_regulator_lipids + self.parameter_dict.basal_rate_regulator_lipids / (
                            1 + ((self.N_regulator_lipids[n_step - 1] / self.volume[
                        n_step - 1]) / self.parameter_dict.michaelis_const_regulator_lipids) ** self.parameter_dict.hill_coeff_regulator_lipids) * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_lipid_regulation == 'al':
                    dn_lipids = dn_lipids + self.parameter_dict.basal_rate_lipids / (
                            1 + ((self.N_lipids[n_step - 1] / self.volume[
                        n_step - 1]) / self.parameter_dict.michaelis_const_lipids) ** self.parameter_dict.hill_coeff_lipids) * self.parameter_dict.time_step
                elif self.parameter_dict.version_of_lipid_regulation == 'constit':
                    dn_lipids = dn_lipids + self.parameter_dict.basal_rate_lipids * self.parameter_dict.time_step

        if self.parameter_dict.version_of_lipid_regulation == 'prop_to_v':
            dn_lipids = dn_lipids + self.parameter_dict.basal_rate_lipids * self.parameter_dict.time_step * self.volume[
                n_step - 1] ** (1)
        self.N_lipids[n_step] = self.N_lipids[n_step - 1] + dn_lipids
        self.N_regulator_lipids[n_step] = self.N_regulator_lipids[n_step - 1] + dn_regulator_lipids
        if self.parameter_dict.version_of_lipid_regulation == 'proteome_sector':
            next_lipid_conc = self.lipid_conc[n_step - 1] \
                              + ( self.parameter_dict.activation_rate_lipids - self.parameter_dict.relaxation_rate * self.lipid_conc[n_step - 1]) * self.parameter_dict.time_step \
                              + np.random.normal(0, 1) * np.sqrt(self.parameter_dict.time_step * 2 * self.parameter_dict.noise_strength_lipids)
            if next_lipid_conc >= 0:
                self.lipid_conc[n_step] = next_lipid_conc
            else:
                self.lipid_conc[n_step] = 0

    def update_sites(self, n_step):
        sum_sites = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                sum_sites = sum_sites + self.chromosome_dict[node_i].sites[-1]
        self.sites_total[n_step] = sum_sites

    def update_length(self, n_step):
        sum_length = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                sum_length = sum_length + self.chromosome_dict[node_i].length[-1]
        self.length_total[n_step] = sum_length

    def count_n_oris(self):
        n_oris = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                n_oris = n_oris + 1
        return n_oris


    def replicate_chromosomes(self, time, n_step):
        """ All chromosomes in the cell that are currently being replicated, are replicated in this step.
            We loop over the chromosome tree and check whether the depth of the tree is the maximal depth.
            If a chromosome is a leave of the tree, we take the chromosome via the corresponding chromosome_dict
            and apply the replication method of this instance of the Chromosome class.

        Parameters
        ----------
        time : double
            The time of the current simulation step
        n_step : double
            The nth step of the simulation
        """
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                self.chromosome_dict[node_i].replicate(time)
        self.update_sites(n_step)
        self.update_length(n_step)

    def calculate_free_dnaA_concentration(self, n_step):
        """ Calculates the free initiator concentration for a given volume, total number of sites and total number
            of initiators. The formula for this is obtained via the law of mass action and specified in the SI of
            the paper.

        Parameters
        ----------
        n_step : double
            The nth step of the simulation
        """
        sum = self.parameter_dict.diss_constant_sites + self.sites_total[n_step - 1] / self.volume[n_step - 1] + \
              self.N_init[
                  n_step - 1] / self.volume[n_step - 1]
        self.free_conc[n_step] = self.N_init[n_step - 1] / self.volume[n_step - 1] - (sum) / 2 + np.sqrt(
            sum ** 2 - 4 * self.sites_total[n_step - 1] / self.volume[n_step - 1] * self.N_init[n_step - 1] /
            self.volume[n_step - 1]) / 2
        self.N_free[n_step] = self.free_conc[n_step] * self.volume[n_step - 1]

    def store_initiation_event(self, n_step):
        self.t_init.append(self.time[n_step])
        self.n_init.append(self.N_init[n_step])
        self.v_init.append(self.volume[n_step])
        self.n_sites_init.append(self.sites_total[n_step])
        self.n_ori_init.append(self.n_ori[n_step - 1])
        self.v_init_per_ori.append(self.volume[n_step] / self.n_ori[n_step - 1])
        self.c_free_init.append(self.free_conc[n_step])
        self.frac_active_init.append(self.active_fraction[n_step])
        self.n_active_init.append(self.N_active[n_step])
        self.lipid_conc_init.append(self.lipid_conc[n_step])
        try:
            self.v_b_before_init.append(self.v_b[-1])
        except:
            self.v_b_before_init.append(0)
            self.print_verbose('last birth volume could not be obtained, this is v_b: {}'.format(self.v_b))

    def initiate_replication(self, n_step, time, volume):
        self.print_verbose('Replication is initiated: critical concentration is reached {}'.format(self.active_conc[n_step]))
        self.store_initiation_event(n_step)
        self.n_ori[n_step] = 2 * self.n_ori[n_step - 1]
        self.N_c_max = 2 * self.N_c_max
        tree_copy = Tree(self.chromosome_tree.subtree(self.chromosome_tree.root), deep=True)
        if self.parameter_dict.verbose == 1:
            tree_copy.show()
        n_chrom = 0
        depth_tree = tree_copy.depth()
        new_generation_depth = depth_tree + 1
        for node_i in tree_copy.expand_tree(mode=2):
            if tree_copy.depth(node_i) == depth_tree:
                for sibling in range(0, 2):
                    name_i = "chromosome " + str(new_generation_depth) + str(n_chrom)
                    self.chromosome_tree.create_node("Chromosome " + str(new_generation_depth) + str(n_chrom), name_i,
                                                     parent=node_i)
                    self.print_verbose('depth of {} :'.format(name_i, self.chromosome_tree.depth(name_i)))
                    if sibling == 0:  # equal to parent chromosome, but new initiation time
                        chromosome = self.chromosome_dict[node_i]
                        chromosome.set_t_end_blocked(time + self.parameter_dict.period_blocked)
                        chromosome.set_t_end_blocked_production(time + self.parameter_dict.period_blocked_production)
                        blocked = chromosome.check_blocked(time)
                        blocked_production = chromosome.check_blocked_production(time)
                        self.print_verbose('now blocked: {} and blocked production: {}'.format(blocked, blocked_production))
                    else:  # active replicating chromosome, length 0
                        chromosome = Chromosome(self.parameter_dict,
                                                t_end_blocked=time + self.parameter_dict.period_blocked,
                                                t_end_blocked_production = time + self.parameter_dict.period_blocked_production,
                                                t_init=time, v_init=volume)
                        blocked = chromosome.check_blocked(time)
                        blocked_production = chromosome.check_blocked_production(time)
                        self.print_verbose('now blocked: {} and blocked production: {}'.format(blocked, blocked_production))
                    self.chromosome_dict.update({name_i: chromosome})
                    self.print_verbose('length, active and blocked of created chromosome: {} {} {}'.format(self.chromosome_dict[name_i].length[-1],
                          self.chromosome_dict[name_i].active_replication, self.chromosome_dict[name_i].blocked))
                    n_chrom = n_chrom + 1
        if self.parameter_dict.verbose == 1:
            self.chromosome_tree.show()

    def find_next_div_time(self, n_step):
        next_div_time = self.parameter_dict.t_max
        list_all_nodes = self.chromosome_tree.all_nodes()
        for i in range(0, len(list_all_nodes)):
            node_i = list_all_nodes[i].identifier
            if self.chromosome_dict[node_i].division_time < next_div_time:
                next_div_time = self.chromosome_dict[node_i].division_time
        if self.parameter_dict.independent_division_cycle == 1:
            if self.volume[n_step] >= self.next_division_volume and next_div_time < self.parameter_dict.t_max:
                next_div_time = self.time[n_step]
            # if replication has finished, but division volume has not been reached yet, cell should not divide yet
            else:
                next_div_time = self.parameter_dict.t_max
        return next_div_time

    def store_chromosomes(self, tree):
        for node_i in tree.expand_tree(mode=2):
            if tree.depth(node_i) == tree.depth():
                self.stored_lengths.append(self.chromosome_dict[node_i].length)
                self.stored_sites.append(self.chromosome_dict[node_i].sites)
                self.stored_times.append(self.chromosome_dict[node_i].time)
            del self.chromosome_dict[node_i]

    def return_division_position_error(self):
        if self.parameter_dict.cv_division_position == 0:
            return 0
        else:
            return np.random.normal(0, self.parameter_dict.cv_division_position)

    def return_rand_binomial_distributed_number(self, prob, n_tot):
        k_rand = np.random.binomial(n_tot, prob)
        return k_rand

    def save_tuple_v_b_v_d(self, n_step):
        if self.last_v_b is None:
            return
        else:
            self.list_of_tuples_v_b_v_d.append((self.last_v_b, self.volume[n_step], self.time[n_step]))

    def divide_volume(self, n_step):
        rel_div_position_error = 0
        if self.parameter_dict.cv_division_position == 0:
            self.volume[n_step] = self.volume[n_step] / 2
        else:
            if self.parameter_dict.single_division_error == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    rel_div_position_error = + self.parameter_dict.cv_division_position
                    self.volume[n_step] = self.volume[n_step] / 2 + rel_div_position_error * self.volume[n_step]
                else:
                    self.volume[n_step] = self.volume[n_step] / 2
            else:
                rel_div_position_error = self.return_division_position_error()
                self.volume[n_step] = self.volume[n_step] / 2 + rel_div_position_error * self.volume[n_step]
        return rel_div_position_error

    def divide_n_initiator(self, n_step, relative_division_error, prob_of_one_protein_being_in_new_cell):
        if self.parameter_dict.partitionning_error_initiator == 0:
            self.N_init[n_step] = self.N_init[n_step] / 2 + relative_division_error * self.N_init[n_step]
        else:
            if self.parameter_dict.single_division_error == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    self.N_init[n_step] = self.N_init[n_step] / 2 + relative_division_error * self.N_init[n_step] \
                                          + self.parameter_dict.single_protein_number_error * self.N_init[n_step]
                else:
                    self.N_init[n_step] = self.N_init[n_step] / 2 + relative_division_error * self.N_init[n_step]
            else:
                self.N_init[n_step] = self.return_rand_binomial_distributed_number(
                    prob_of_one_protein_being_in_new_cell, self.N_init[n_step])

    def divide_n_regulator(self, n_step, relative_division_error, prob_of_one_protein_being_in_new_cell):
        if self.parameter_dict.partitionning_error_regulator == 0:
            self.N_regulator[n_step] = self.N_regulator[n_step] / 2 + relative_division_error * self.N_regulator[n_step]
        else:
            if self.parameter_dict.single_division_error == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    self.N_regulator[n_step] = self.N_regulator[n_step] / 2 + relative_division_error * \
                                               self.N_regulator[n_step] \
                                               - self.parameter_dict.single_protein_number_error * self.N_regulator[
                                                   n_step]
                else:
                    self.N_regulator[n_step] = self.N_regulator[n_step] / 2 + relative_division_error * \
                                               self.N_regulator[n_step]
            else:
                self.N_regulator[n_step] = self.return_rand_binomial_distributed_number(
                    prob_of_one_protein_being_in_new_cell, self.N_regulator[n_step])


    def divide_n_active_initiator(self, n_step, relative_division_error, prob_of_one_protein_being_in_new_cell):
        if self.parameter_dict.partitionning_error_initiator == 0:
            self.N_active[n_step] = self.N_active[n_step] / 2 + relative_division_error * self.N_active[n_step]
        else:
            if self.parameter_dict.single_division_error == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    self.N_active[n_step] = self.N_active[n_step] / 2 + relative_division_error * self.N_active[n_step] \
                                          + self.parameter_dict.single_protein_number_error * self.N_active[n_step]
                else:
                    self.N_active[n_step] = self.N_active[n_step] / 2 + relative_division_error * self.N_active[n_step]
            else:
                self.N_active[n_step] = self.return_rand_binomial_distributed_number(
                    prob_of_one_protein_being_in_new_cell, self.N_active[n_step])

    def divide_lipids(self, n_step, relative_division_error, prob_of_one_protein_being_in_new_cell):
        if self.parameter_dict.partitionning_error_lipids == 0:
            self.N_lipids[n_step] = self.N_lipids[n_step] / 2 + relative_division_error * self.N_lipids[n_step]
        else:
            if self.parameter_dict.single_division_error_lipids == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    self.N_lipids[n_step] = self.N_lipids[n_step] / 2 + relative_division_error * self.N_lipids[n_step] \
                                            + self.parameter_dict.single_protein_number_error * self.N_lipids[n_step]
                else:
                    self.N_lipids[n_step] = self.N_lipids[n_step] / 2 + relative_division_error * self.N_lipids[n_step]
            else:
                self.N_lipids[n_step] = self.return_rand_binomial_distributed_number(
                    prob_of_one_protein_being_in_new_cell, self.N_lipids[n_step])
        # if self.parameter_dict.model_lipids_explicitly == 1:
        #     if len(self.t_div) == self.parameter_dict.cycle_with_error:
        #         self.lipid_conc[n_step] = self.parameter_dict.lipid_conc_0

    def divide_regulator_lipids(self, n_step, relative_division_error, prob_of_one_protein_being_in_new_cell):
        if self.parameter_dict.partitionning_error_regulator_lipids == 0:
            self.N_regulator_lipids[n_step] = self.N_regulator_lipids[n_step] / 2 + relative_division_error * \
                                              self.N_regulator_lipids[n_step]
        else:
            if self.parameter_dict.single_division_error_regulator_lipids == 1:
                if len(self.t_div) == self.parameter_dict.cycle_with_error:
                    self.N_regulator_lipids[n_step] = self.N_regulator_lipids[n_step] / 2 + relative_division_error * \
                                                      self.N_regulator_lipids[n_step] \
                                                      - self.parameter_dict.single_protein_number_error * \
                                                      self.N_regulator_lipids[n_step]
                else:
                    self.N_regulator_lipids[n_step] = self.N_regulator_lipids[n_step] / 2 + relative_division_error * \
                                                      self.N_regulator_lipids[n_step]
            else:
                self.N_regulator_lipids[n_step] = self.return_rand_binomial_distributed_number(
                    prob_of_one_protein_being_in_new_cell, self.N_regulator_lipids[n_step])

    def divide_variables(self, n_step):
        self.save_tuple_v_b_v_d(n_step)
        volume_mother_at_division = self.volume[n_step]

        # now divide volume with or without error
        rel_div_position_error = self.divide_volume(n_step)
        self.v_b = np.append(self.v_b, self.volume[n_step])
        self.last_v_b = self.volume[n_step]

        # divide chromosome exactly by two always
        self.n_ori[n_step] = self.n_ori[n_step] / 2
        self.N_c_max = self.N_c_max / 2
        self.sites_total[n_step] = self.sites_total[n_step] / 2
        self.length_total[n_step] = self.length_total[n_step] / 2
        # self.N_init_ext[n_step] = self.N_init_ext[n_step] / 2

        prob_of_one_protein_being_in_new_cell = (self.volume[n_step]) / volume_mother_at_division
        # divide number of initiator proteins
        self.divide_n_initiator(n_step, rel_div_position_error, prob_of_one_protein_being_in_new_cell)
        # divide number of regulator proteins
        self.divide_n_regulator(n_step, rel_div_position_error, prob_of_one_protein_being_in_new_cell)
        # divide number of active initiator proteins
        self.divide_n_active_initiator(n_step, rel_div_position_error, prob_of_one_protein_being_in_new_cell)

        # divide lipids
        self.divide_lipids(n_step, rel_div_position_error, prob_of_one_protein_being_in_new_cell)
        self.divide_regulator_lipids(n_step, rel_div_position_error, prob_of_one_protein_being_in_new_cell)

    def returns_chromosome_tree(self, depth):
        new_tree = Tree()
        new_tree.create_node('Chromosome 0', 'chromosome 0')
        if depth == 0:
            return new_tree
        new_tree.create_node('Chromosome 10', 'chromosome 10', parent='chromosome 0')
        new_tree.create_node('Chromosome 11', 'chromosome 11', parent='chromosome 0')
        if depth == 1:
            return new_tree
        new_tree.create_node('Chromosome 20', 'chromosome 20', parent='chromosome 10')
        new_tree.create_node('Chromosome 21', 'chromosome 21', parent='chromosome 10')
        new_tree.create_node('Chromosome 22', 'chromosome 22', parent='chromosome 11')
        new_tree.create_node('Chromosome 23', 'chromosome 23', parent='chromosome 11')
        if depth == 2:
            return new_tree
        new_tree.create_node('Chromosome 30', 'chromosome 30', parent='chromosome 20')
        new_tree.create_node('Chromosome 31', 'chromosome 31', parent='chromosome 20')
        new_tree.create_node('Chromosome 32', 'chromosome 32', parent='chromosome 21')
        new_tree.create_node('Chromosome 33', 'chromosome 33', parent='chromosome 21')
        new_tree.create_node('Chromosome 34', 'chromosome 34', parent='chromosome 22')
        new_tree.create_node('Chromosome 35', 'chromosome 35', parent='chromosome 22')
        new_tree.create_node('Chromosome 36', 'chromosome 36', parent='chromosome 23')
        new_tree.create_node('Chromosome 37', 'chromosome 37', parent='chromosome 23')
        if depth == 3:
            return new_tree
        new_tree.create_node('Chromosome 40', 'chromosome 40', parent='chromosome 30')
        new_tree.create_node('Chromosome 41', 'chromosome 41', parent='chromosome 30')
        new_tree.create_node('Chromosome 42', 'chromosome 42', parent='chromosome 31')
        new_tree.create_node('Chromosome 43', 'chromosome 43', parent='chromosome 31')
        new_tree.create_node('Chromosome 44', 'chromosome 44', parent='chromosome 32')
        new_tree.create_node('Chromosome 45', 'chromosome 45', parent='chromosome 32')
        new_tree.create_node('Chromosome 46', 'chromosome 46', parent='chromosome 33')
        new_tree.create_node('Chromosome 47', 'chromosome 47', parent='chromosome 33')
        new_tree.create_node('Chromosome 48', 'chromosome 48', parent='chromosome 34')
        new_tree.create_node('Chromosome 49', 'chromosome 49', parent='chromosome 34')
        new_tree.create_node('Chromosome 50', 'chromosome 50', parent='chromosome 35')
        new_tree.create_node('Chromosome 51', 'chromosome 51', parent='chromosome 35')
        new_tree.create_node('Chromosome 52', 'chromosome 52', parent='chromosome 36')
        new_tree.create_node('Chromosome 53', 'chromosome 53', parent='chromosome 36')
        new_tree.create_node('Chromosome 54', 'chromosome 54', parent='chromosome 37')
        new_tree.create_node('Chromosome 55', 'chromosome 55', parent='chromosome 37')
        if depth > 3:
            print('depth of chromosome tree after division was bigger than 3')
            self.abort_simulation = True
            return new_tree
            # exit()

    def update_dictionary(self, surviving_tree, new_tree):
        dict_old_new_keys = {'chromosome 10': 'chromosome 0', 'chromosome 20': 'chromosome 10',
                             'chromosome 21': 'chromosome 11',
                             'chromosome 30': 'chromosome 20', 'chromosome 31': 'chromosome 21',
                             'chromosome 32': 'chromosome 22', 'chromosome 33': 'chromosome 23',
                             'chromosome 40': 'chromosome 30', 'chromosome 41': 'chromosome 31',
                             'chromosome 42': 'chromosome 32', 'chromosome 43': 'chromosome 33',
                             'chromosome 44': 'chromosome 34', 'chromosome 45': 'chromosome 35',
                             'chromosome 46': 'chromosome 36', 'chromosome 47': 'chromosome 37'}
        for node_i in surviving_tree.expand_tree(mode=2):
            # print('old key, new key: ', node_i, dict_old_new_keys[node_i])
            self.chromosome_dict[dict_old_new_keys[node_i]] = self.chromosome_dict.pop(node_i)

    def calculate_new_division_volume(self):
        division_volume = None
        if self.parameter_dict.version_of_independent_division_regulation == 'sizer':
            division_volume = self.parameter_dict.division_volume
        if self.parameter_dict.version_of_independent_division_regulation == 'IDA':
            division_volume = self.last_v_b + self.calculate_noisy_added_volume()
        return division_volume

    def calculate_noisy_added_volume(self):
        return self.parameter_dict.division_volume/2 + np.random.normal(0, self.parameter_dict.cv_added_volume) * self.parameter_dict.division_volume/2

    def divide(self, n_step):
        self.divide_variables(n_step)
        self.t_div.append(self.time[n_step])
        #  if separate division control:
        if self.parameter_dict.independent_division_cycle == 1:
            self.next_division_volume = self.calculate_new_division_volume()
        #  split tree in two
        if self.parameter_dict.verbose == 1:
            self.chromosome_tree.show()
        surviving_tree = self.chromosome_tree.subtree('chromosome 10')
        deleted_tree = self.chromosome_tree.subtree('chromosome 11')
        self.print_verbose('division time: {} and new tree after division {}'.format(self.time[n_step], self.chromosome_dict[surviving_tree.root].length))
        if self.parameter_dict.verbose == 1:
            surviving_tree.show()
            deleted_tree.show()
        # store and delete second half of tree, delete root
        self.store_chromosomes(deleted_tree)
        del self.chromosome_dict['chromosome 0']
        self.print_verbose('dict after deletion of root: {}'.format(self.chromosome_dict))
        # rename dictionary and nodes in tree
        self.print_verbose('Depth of tree: {}'.format(surviving_tree.depth()))
        new_tree = self.returns_chromosome_tree(surviving_tree.depth())
        if self.abort_simulation == 1:
            return
        if self.parameter_dict.verbose == 1:
            surviving_tree.show()
            new_tree.show()
        self.update_dictionary(surviving_tree, new_tree)
        self.chromosome_tree = new_tree

    # From ultrasensitivity
    def calculate_active_concentration_and_fraction(self, n_step):
        if self.parameter_dict.intrinsic_switch_noise == 1:
            intrinsic_switch_noise = np.random.normal(0, 1) * np.sqrt(self.parameter_dict.time_step * 2 * self.parameter_dict.noise_strength_switch)
        else:
            intrinsic_switch_noise = 0
        self.total_conc[n_step] = self.parameter_dict.total_conc
        self.active_conc[n_step] = self.active_conc[n_step - 1] + (
                self.calculate_activation_rate(n_step, self.total_conc[n_step])
                - self.calculate_deactivation_rate(n_step)
                + self.calculate_production_rate_synthesis(
            n_step, self.total_conc[n_step])) * self.parameter_dict.time_step + intrinsic_switch_noise
        if self.active_conc[n_step] < 0:
            self.active_conc[n_step] = 0
        self.active_fraction[n_step] = self.active_conc[n_step] / self.total_conc[n_step]

    # From ultrasensitivity
    def calculate_active_concentration_and_fraction_init_explicitly_expressed(self, n_step, noise_dict):
        # the main difference to the non-explicit version is that here the change in the active number of DnaA includes the noise production term from the total DnaA production
        if self.parameter_dict.intrinsic_switch_noise == 1:
            intrinsic_switch_noise = np.random.normal(0, 1) * np.sqrt(self.parameter_dict.time_step * 2 * self.parameter_dict.noise_strength_switch)
        else:
            intrinsic_switch_noise = 0
        self.total_conc[n_step] = self.N_init[n_step - 1] / self.volume[n_step - 1]
        self.N_active[n_step] = self.N_active[n_step - 1] + (
                self.calculate_activation_rate(n_step, self.total_conc[n_step])
                - self.calculate_deactivation_rate(n_step)) * self.volume[n_step - 1] \
                                * self.parameter_dict.time_step \
                                + noise_dict['dn_init'] \
                                + noise_dict['noise_init']
        if self.N_active[n_step] < 0:
            self.N_active[n_step] = 0
        self.active_conc[n_step] = self.N_active[n_step] / self.volume[n_step - 1]+ intrinsic_switch_noise
        self.active_fraction[n_step] = self.active_conc[n_step] / self.total_conc[n_step]

    def calculate_activation_rate(self, n_step, total_conc):
        rate_0 = 0
        rate_1 = 0
        aspect_ratio_v = 1
        oric_factor = 1
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                rate_0 = rate_0 + self.chromosome_dict[node_i].n_dars2 * self.parameter_dict.activation_rate_dars2 + \
                         self.chromosome_dict[node_i].n_dars2_high_activity * self.parameter_dict.high_rate_dars2
                rate_1 = rate_1 + self.chromosome_dict[node_i].n_dars1 * self.parameter_dict.activation_rate_dars1
        self.activation_rate_dars2_tot[n_step] = rate_0
        self.activation_rate_dars1_tot[n_step] = rate_1
        if self.parameter_dict.surface_area_to_vol_const == 0:
            aspect_ratio_v = 2 * (self.parameter_dict.aspect_ratio + 1) / self.parameter_dict.aspect_ratio * (
                    self.parameter_dict.aspect_ratio * np.pi / self.volume[n_step - 1]) ** (1 / 3) / 5.47
        if self.parameter_dict.lipid_oric_dependent ==1:
            oric_factor = self.n_ori[n_step - 1] / self.volume[n_step - 1]
        if self.parameter_dict.model_lipids_explicitly == 0:
            activation_rate_lipids = self.parameter_dict.activation_rate_lipids * aspect_ratio_v * oric_factor
        else:
            if self.parameter_dict.version_of_lipid_regulation == 'proteome_sector':
                activation_rate_lipids = self.lipid_conc[n_step - 1] * aspect_ratio_v * oric_factor
            else:
                activation_rate_lipids = self.N_lipids[n_step - 1] / self.volume[n_step - 1] * aspect_ratio_v * oric_factor
        self.activation_rate_lipids_tot[n_step] = activation_rate_lipids
        rate = (activation_rate_lipids * aspect_ratio_v + (rate_0 + rate_1) / self.volume[n_step - 1]) * (
                total_conc - self.active_conc[n_step - 1]) / (
                       self.parameter_dict.michaelis_const_prod + total_conc - self.active_conc[n_step - 1])
        return rate

    def update_rida_deactivation_rate(self, n_steps):
        if self.parameter_dict.model_rida_explicitly == 0:
            self.rida_deactivation_rate[n_steps] = self.parameter_dict.deactivation_rate_rida
        else:
            self.rida_noise[n_steps] = self.rida_noise[n_steps -1] - self.parameter_dict.rate_growth * self.rida_noise[n_steps-1] * self.parameter_dict.time_step \
            + np.random.normal(0, 1) * np.sqrt(self.parameter_dict.time_step * 2 * self.parameter_dict.noise_strength_rida)
            self.rida_deactivation_rate[n_steps] = self.parameter_dict.deactivation_rate_rida + self.rida_noise[n_steps]

    def calculate_deactivation_rate(self, n_step):
        self.update_rida_deactivation_rate(n_step)
        rate_datA = 0
        rate_rida = 0
        for node_i in self.chromosome_tree.expand_tree(mode=2):
            if self.chromosome_tree.depth(node_i) == self.chromosome_tree.depth():
                rate_datA = rate_datA + self.chromosome_dict[
                    node_i].n_datA * self.parameter_dict.deactivation_rate_datA + self.chromosome_dict[
                                node_i].n_datA_high_activity * self.parameter_dict.high_rate_datA
                rate_rida = rate_rida + self.chromosome_dict[node_i].n_rida * self.rida_deactivation_rate[n_step]
        self.deactivation_rate_datA_tot[n_step] = rate_datA
        self.deactivation_rate_rida_tot[n_step] = rate_rida
        rate = (rate_datA + rate_rida) / self.volume[n_step - 1] * self.active_conc[n_step - 1] / (
                self.parameter_dict.michaelis_const_destr + self.active_conc[n_step - 1])
        return rate

    def calculate_production_rate_synthesis(self, n_step, total_conc):
        if self.parameter_dict.include_synthesis == 1:
            rate = self.parameter_dict.rate_growth * (total_conc - self.active_conc[n_step - 1])
        else:
            rate = 0
        return rate

    def create_one_cell_cycle_volume_array(self):
        indx_min = np.where(self.volume == self.v_b[-2])[0][0]
        indx_max = np.where(self.volume == self.v_b[-1])[0][0]
        cell_cycle_volumes = self.volume[indx_min:indx_max]
        cell_cycle_n_oris = self.n_ori[indx_min:indx_max]
        cell_cycle_conc = self.active_conc[indx_min:indx_max]
        size_final = 15
        dstep = int(cell_cycle_volumes.size / size_final)
        cell_cycle_volumes_short = cell_cycle_volumes[1::dstep]
        cell_cycle_n_oris_short = cell_cycle_n_oris[1::dstep]
        cell_cycle_conc_short = cell_cycle_conc[1::dstep]
        return cell_cycle_volumes_short, cell_cycle_n_oris_short, cell_cycle_conc_short

    def plot_rates_together(self, filepath, volumes=np.array([]), n_oris=np.array([])):
        concentrations = np.arange(0, self.parameter_dict.total_conc, 0.01)
        series_rates = []
        series_conc = []
        if volumes.size == 0:
            volumes, n_oris, conc = self.create_one_cell_cycle_volume_array()
            self.min_frac = np.amin(conc) / self.parameter_dict.total_conc
        rate_production = self.calculate_activation_rate(concentrations)
        series_rates.append(rate_production)
        series_conc.append(concentrations)
        for item in range(0, volumes.size):
            rate_deactivation = self.calculate_deactivation_rate(volumes[item], n_oris[item], concentrations)
            series_rates.append(rate_deactivation)
            series_conc.append(concentrations)
            rate_production = self.calculate_activation_rate(volumes[item], n_oris[item], concentrations)
            series_rates.append(rate_production)
            series_conc.append(concentrations)
        plottingTools.plot_series_of_one_array(filepath, series_conc, series_rates, r'[DnaA-ATP]', 'rates',
                                               'rate_comparison')

    def determine_min_frac(self):
        volumes, n_oris, conc = self.create_one_cell_cycle_volume_array()
        self.min_frac = np.amin(conc) / self.parameter_dict.total_conc

    def decide_whether_initiate_rep(self, n_step):
        if self.parameter_dict.version_of_model == 'switch':
            return self.active_conc[n_step] > self.parameter_dict.init_conc and self.time[n_step] > self.t_end_blocked
        if self.parameter_dict.version_of_model == 'titration':
            return self.free_conc[n_step] > self.parameter_dict.critical_free_conc and self.time[
                n_step] > self.t_end_blocked
        if self.parameter_dict.version_of_model == 'switch_titration':
            return self.free_conc[n_step] * self.active_fraction[
                n_step] > self.parameter_dict.critical_free_active_conc and self.time[n_step] > self.t_end_blocked
        if self.parameter_dict.version_of_model == 'switch_critical_frac':
            return self.active_fraction[n_step] > self.parameter_dict.frac_init and self.time[
                n_step] > self.t_end_blocked

    def check_whether_once_lipid_conc_perturb(self, n_step):
        if self.parameter_dict.single_lipid_conc_perturb==1 and self.one_time_perturbation == 0 and self.time[n_step] >= self.parameter_dict.time_of_perturb:
            self.lipid_conc[n_step] = self.parameter_dict.lipid_conc_0
            self.one_time_perturbation = 1

    def run_cell_cycle(self):
        """ This is the main function of this class and runs the cell cycle by updating all variables for every time step. """
        for n_step in range(0, self.time.size):
            self.grow_cell(n_step)  # update cell volume
            self.replicate_chromosomes(self.time[n_step], n_step)  # replicate chromosomes
            self.calculate_free_dnaA_concentration(n_step)
            noise_dict = self.produce_repressed_dnaA_deterministic(n_step, self.time[n_step])
            if self.parameter_dict.model_lipids_explicitly == 1:
                self.produce_lipids(n_step)
            # compute active fraction
            if self.parameter_dict.initiator_explicitly_expressed == 0:
                self.calculate_active_concentration_and_fraction(n_step)
            else:
                self.calculate_active_concentration_and_fraction_init_explicitly_expressed(n_step, noise_dict)
            if self.decide_whether_initiate_rep(n_step):
                self.initiate_replication(n_step, self.time[n_step], self.volume[n_step])
                self.t_end_blocked = self.time[n_step] + self.parameter_dict.period_blocked
                self.print_verbose('new t end blocked: {}'.format(self.t_end_blocked))
            else:
                self.n_ori[n_step] = self.n_ori[n_step - 1]
            self.next_div_t = self.find_next_div_time(n_step)
            if self.time[n_step] >= self.next_div_t:
                self.divide(n_step)
                if self.abort_simulation:
                    print('cell cycle simulation is aborted now')
                    break
                self.calculate_free_dnaA_concentration(n_step)
            self.check_whether_once_lipid_conc_perturb(n_step)

        # self.determine_min_frac()
        self.makeDataFrameOfCellCycle()
        return
