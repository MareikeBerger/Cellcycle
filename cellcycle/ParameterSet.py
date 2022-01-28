from tkinter import E
import numpy as np
import pandas as pd
import git


class ParameterSet:
    """This class is used to generate the parameter set of a simulation."""
    def __init__(self):
        """Generates the parameter set of a simulation as soon as class is initiated."""
        # Parameters specifying the series number of the simulation
        self.n_series = 3 # total number of series
        self.id = np.arange(self.n_series) # makes an individual id for each of the n_series

        # Storing git version and commit identifier of code
        try:
            self.git_version = git.Repo('.', search_parent_directories=True).head.object.hexsha
        except:
            print('no git repository for storing git SHA')
        
        # Parameters of simulation
        self.doubling_rate = 0.5 * np.ones(self.n_series) # in units 1/h, default parameters [low: 0.5, int: 60/35, high: 60/25, max:2.5]
        self.n_cycles = 20 # number of cell cycles the simulation should do [default: 20]
        self.t_max = self.n_cycles / self.doubling_rate  # maximal time of simulation in units of hours, calculated by dividing the desired number of cycles by the time per cycle
        self.period_blocked = 0.17 * np.ones(self.n_series)  # time in hours during which no new DnaA is produced and/or oriC is blocked for initiation [default: 0.17] (is approx 10 minutes)
        self.t_C = 2 / 3 * np.ones(self.n_series)  # time in hours that it takes to replicate the chromosome, [default: 2/3] (=40 min)
        self.t_D = 1 / 3 * np.ones(self.n_series)  # time in hours that it takes from the end of replication until division, [default: 1/3] (=20 min)
        self.t_CD = self.t_C + self.t_D  # time in hours that it takes from end of replication until division, [default: 1] (=60 min)
        self.rate_growth = self.doubling_rate * np.log(2)  # growth rate of the cell
        self.time_step = 0.001 * np.ones(self.n_series)  # time step in h of simulation time, [default: 0.001], choose such that time_step < 1/maximal_rate_of_simulation
        
        # Paramaters specifying initial conditions
        self.n_ori_0 = np.ones(self.n_series)  # number of origins at beginning, [default: 1]
        self.v_0 = 0.1 * np.ones(self.n_series)  # volume at beginning in units of micro meters, [default: 0.1]

        # version of the model, determines parameters later
        self.version_of_model = 'switch_titration'    
                                            # "titration"-> replication initiation is triggered by concentration of initiator proteins in cytosol
                                            # "switch"-> replication initiation is triggered when critical concentration of total concentration is reached
                                            # "switch_critical_frac" -> replication initiation is triggered at the critical fraction of ATP-DnaA
                                            # "switch_titration" -> replication initiation is triggered when concentration of active form of initiator proteins in the cytosol reaches critical concentration
        self.initiator_explicitly_expressed = 1  # very important parameter, if one, then N_init is explicitly expressed, if 0 then total concentration is simply constant and given by total_conc
        self.version_of_titration = 'regulator_is_initiator' # specifies how initiator protein is expressed [default: 'regulator_is_initiator']
                                            # different versions of titration mechanism:
                                            #  'regulator_is_initiator' -> initiator protein is negatively autoregulated, there is no separate regulator protein
                                            #  'regulator_is_not_initiator' -> two proteins, one regulator and one initiator protein
                                            #  'regulator_and_init_constit_expressed' -> testing behavior of constit expressed gene
                                            #  'regulator_and_init_prop_to_v' -> testing behavior of proportional to volume
                                            #  'different_promoters' -> testing effect of having two promoters, one for ATP-DnaA and one for ADP-DnaA
        self.gene_expression_model = 'ribo_limiting' # specifies which gene expression model is being used [default: 'ribo_limiting']
                                            # different gene expression models could be used:
                                            # 'ribo_limiting' the basal protein production rate depends explicitly on the volume following the growing cell model of gene expression (see SI)
                                            # 'standard' the basal protein production rate is constant following the traditional gene expression model (see SI)
        self.lddr = 1  # if 1 then we use the LDDR model, otherwise parameters of LD model

        # Parameters for switch model
        self.michaelis_const_destr = 50 * np.ones(self.n_series)  # Michaelis constant for DnaA deactivators in units of number per volume [default: 50], for ultra-sensitive regime 5
        self.michaelis_const_prod = 50 * np.ones(self.n_series)  # Michaelis constant for DnaA activators in units of number per volume [default: 50], for ultra-sensitive regime 5
        self.total_conc = 400 * np.ones(self.n_series)  # total DnaA concentration in units of number per volume (cubic micrometers^-1) [default: 400]
        self.frac_init = 0.75 * np.ones(self.n_series)  # critical fraction of ATP-DnaA at which replication is initiated at all origins simultaneously [default: 0.75] 
        if self.version_of_model == 'switch_titration':
            self.init_conc = 200 * np.ones(self.n_series) # critical ATP-DnaA concentration at which replication is initiated at all origins simultaneously [default: 200]
        else:
            self.init_conc = 300 * np.ones(self.n_series) # critical ATP-DnaA concentration at which replication is initiated at all origins simultaneously [default: 300]
        if self.version_of_model == 'titration' or self.version_of_model == 'switch_titration':
            self.conc_0 = 1000 * np.ones(self.n_series) # total DnaA concentration at the beginning of the simulation (initial condition) [default: 1000]
        else:
            self.conc_0 = 400 * np.ones(self.n_series) # total DnaA concentration at the beginning of the simulation (initial condition) [default: 400]
        
        # Parameters for location of DARS1/2 on the chromosome
        self.t_doubling_dars2 = 0.2 * np.ones(self.n_series)  # time from replication initiation to replication of DARS2 in units of hours [default: 0.2] (=12 min)
        self.site_dars2 = self.t_doubling_dars2 / self.t_C # relative position of DARS2 on the chromosome
        self.t_doubling_dars1 = 0.1 * np.ones(self.n_series)   # time from replication initiation to replication of DARS1 in units of hours [default: 0.1] (=6 min)
        self.site_dars1 = self.t_doubling_dars1 / self.t_C # relative position of DARS2 on the chromosome

        # Parameters for (de)activation rates in LD/LDDR model, respectively
        if self.lddr == 1:
            self.deactivation_rate_rida = 500 * np.ones(self.n_series)  # RIDA deactivation rate in units per hour [default: 500] (for LDDR model)
            self.deactivation_rate_datA = 300 * np.ones(self.n_series)  # datA deactivation rate (low) in units per hour [default: 300] (for LDDR model)
            self.activation_rate_lipids = 750 * np.ones(self.n_series) # lipid activation rate in units per hour [default: 300] (for LDDR model)
            self.activation_rate_dars2 = 50 * np.ones(self.n_series)  # DARS2 activation rate in units per hour [default: 50] (for LDDR model)
            self.activation_rate_dars1 = 100 * np.ones(self.n_series)  # DARS1 activation rate in units per hour [default: 100] (for LDDR model)
            self.high_rate_datA = 300 * np.ones(self.n_series)  # datA deactivation rate (high) in units per hour [default: 300] (for LDDR model) [total deactivation rate is high_rate_datA+deactivation_rate_datA]
        else:
            self.deactivation_rate_rida = 0 * np.ones(self.n_series)   # RIDA deactivation rate in units per hour [default: 0] (for LD model)
            self.deactivation_rate_datA = 600 * np.ones(self.n_series)  # datA deactivation rate (low) in units per hour [default: 600] (for LD model)
            self.activation_rate_lipids = 2755 * np.ones(self.n_series) # lipid activation rate in units per hour [default: 2755] (for LD model)
            self.activation_rate_dars2 = 0 * np.ones(self.n_series)  # DARS2 activation rate in units per hour [default: 0] (for LD model)
            self.activation_rate_dars1 = 0 * np.ones(self.n_series)  # DARS1 activation rate in units per hour [default: 0] (for LD model)
            self.high_rate_datA = 0 * np.ones(self.n_series)  # datA deactivation rate (high) in units per hour [default: 0] (for LD model)
        self.include_synthesis = 1 * np.ones(self.n_series) # if True, the term \lambda * (1-f) is added when the active fraction is calculated

        # Parameters for cell cycle dependent activation and deactivation rates
        self.t_onset_dars2 = self.t_doubling_dars2  # time in hours from moment of replication initiation to when DARS2 begins being more active [default: t_doubling_dars2]
        self.relative_chromosome_position_onset_dars2 = self.t_onset_dars2 / self.t_C # relative position on the chromosome when DARS2 begins to be more active
        self.t_onset_datA = 0 * np.ones(self.n_series)  # time in hours from moment of replication initiation to when datA begins being more active [default: 0]
        self.relative_chromosome_position_onset_datA = self.t_onset_datA / self.t_C # relative position on the chromosome when DARS2 begins to be more active
        self.t_offset_datA = 0.2 * np.ones(self.n_series)  # time in hours from moment of replication initiation to when datA stops being more active [default: 0.2]
        self.relative_chromosome_position_offset_datA = self.t_offset_datA / self.t_C # relative position on the chromosome when datA stops to be more active
        self.t_offset_dars2 = self.t_C  # time in hours from moment of replication initdars2iation to when DARS2 stops being more active [default: t_c] (=40 min)
        self.relative_chromosome_position_offset_dars2 = self.t_offset_dars2 / self.t_C # relative position on the chromosome when DARS2 stops to be more active
        self.high_rate_dars2 = self.calculate_dars2_from_rida_rate() # activation rate during the high activity time period of DARS2 

        # Parameters for RIT/AIT model
        self.n_init_0 = self.conc_0 * self.v_0 * np.ones(self.n_series)  # number of DnaA at time beginning of simulation
        if self.version_of_model=='titration' or self.version_of_model == 'switch_titration':
            self.n_c_max_0 = 300 * np.ones(self.n_series)  # number of DnaA boxes per chromosome [default: 300]
        else:
            self.n_c_max_0 = 0 * np.ones(self.n_series) # number of DnaA boxes per chromosome [default: 0] in switch model without titration
        self.rate_rep = 1 / self.t_C  # rate at which a chromosome of length 1 is replicated in units 1/h [default: 1/t_C]
        self.rate_synth_sites = self.n_c_max_0 * self.rate_rep  # rate of freeing new sites in 1/h if all titration sites are distributed homogeneously on the chromosome
        self.homogeneous_dist_sites = 1 * np.ones(self.n_series) # if 1 -> sites are distributed homogeneously; if 0 -> all sites are located on the origin
        self.diss_constant_sites = 1 * np.ones(self.n_series) # dissociation constant of DnaA boxes in units per volume (in cubic micrometers^-1) [default: 1]
        self.critical_free_conc = 20 * np.ones(self.n_series) # concentration in cytoplasm at which replication is initiated at all origins [default: 20]

        # Parameters for regulator protein, currently not used
        self.n_regulator_0 = self.n_init_0 # number of initiators at beginning of simulation
        self.michaelis_const_regulator =200 * np.ones(self.n_series)  # dissociation constant of promoter of regulator protein in units per volume (in cubic micrometers^-1) [default: 200]
        self.hill_coeff_regulator = 5 * np.ones(self.n_series)  # hill coefficient of promoter of regulator protein [default:5]
        self.basal_rate_regulator = 70 * np.ones(self.n_series)  #  basal expression rate of regulator protein [default: 70]

        # Parameters for gene expression of initiator protein DnaA
        # if we use two different promotors for ATP- and ADP-DnaA, then these parameters are for ATP-DnaA protein
        if self.version_of_model == 'titration':
            self.michaelis_const_initiator = 200 * np.ones(self.n_series)  # dissociation constant of promoter of (ATP-)DnaA in units per volume (in cubic micrometers^-1) [default: 200]
            self.basal_rate_initiator = 1000 * np.ones(self.n_series)  #  basal (ATP-)DDnaA expression rate [default: 1000]
        if self.version_of_model == 'switch' or self.version_of_model == 'switch_critical_frac':
            self.michaelis_const_initiator = 300 * np.ones(self.n_series)  # dissociation constant of promoter of (ATP-)DDnaA in units per volume (in cubic micrometers^-1) [default: 300]
            self.basal_rate_initiator = 1500 * np.ones(self.n_series)  #  basal (ATP-)DDnaA expression rate [default: 1500]
        if self.version_of_model == 'switch_titration':
            self.michaelis_const_initiator = 400 * np.ones(self.n_series) # dissociation constant of promoter of (ATP-)DDnaA in units per volume (in cubic micrometers^-1) [default: 400]
            self.basal_rate_initiator = 2000 * np.ones(self.n_series)  #  basal (ATP-)DDnaA expression rate [default: 2000]
        self.hill_coeff_initiator =  5 * np.ones(self.n_series)  # hill coefficient of (ATP-)DDnaA promoter [default:5]

        # Parameters for ADP-DnaA protein, if we use two promotors for ATP- and ADP-DnaA, currently not used
        self.michaelis_const_adp_dnaa = 400 * np.ones(self.n_series) # dissociation constant of promoter of ADP-DnaA in units per volume (in cubic micrometers^-1) [default: 400]
        self.hill_coeff_adp_dnaa =  2 * np.ones(self.n_series)  # hill coefficient of ADP-DDnaA promoter [default:2]
        self.cooperativity = 0   # specifies whether DnaA binding to DnaA promoter is cooperative (0->not cooperative, 1 -> cooperative) [default: 0]

        # Parameters for combined switch-titration model
        if self.version_of_model == 'switch_titration':
            self.critical_free_active_conc =  200 * np.ones(self.n_series) # critical free, ATP-DnaA concentration in units per volume (in cubic micrometers^-1) at which replication is initiated [default: 200]
        else:
            self.critical_free_active_conc = self.init_conc

        # Parameters for geometry of the cell, currently not used
        self.surface_area_to_vol_const = 1 * np.ones(self.n_series) # if 1 then surface is directly proportional to volume growth, if not, then surface area to volume is not constant and varies over course of cell cycle [default: 1]
        self.aspect_ratio = 4 * np.ones(self.n_series) # length of cell divided by width of cell as reported by experiments [default: 4]

        # Parameters for modelling lipids explicitly
        self.model_lipids_explicitly = 0  # if 1 -> total number of lipids modelled explicitly, if 0-> then lipid concentration is included in activation rate
        if self.model_lipids_explicitly == 1:
            self.activation_rate_lipids = self.rate_growth * self.activation_rate_lipids # such that lipid concentraiton remains constant as a function of the growth rate
        self.version_of_lipid_regulation = 'proteome_sector'  # different versions of how lipid expression can be regulated: [default: 'proteome_sector'] 
                                                                # "rl" -> regulated lipids by negatively auto-regulated regulator
                                                                # "proteome_sector" -> using an ornstein-uhlenbeck process to include noise in lipid concentration, see equation 6 in paper
                                                                # "al" -> negatively auto-regulated lipids
                                                                # "constit" -> constitutively expressed lipids
                                                                # "prop_to_v" -> rate of lipid production is proportional to cell volume
        self.n_lipids_0 = self.n_init_0 # number of lipids at beginning of simulation [default: n_init_0]
        self.partitionning_error_lipids = 0 * np.ones(self.n_series) # 0 -> no error resulting from unequal lipid partitioning upon cell division, 1-> partitioning error is included [default: 0]
        self.partitionning_error_regulator_lipids = 0 * np.ones(self.n_series) # 0 -> no error resulting from unequal partitioning of the regulator of the lipids upon cell division, 1-> partitioning error of regulator is included [default: 0]
        self.single_division_error_lipids = 0 * np.ones(
            self.n_series)  # either 1 or 0, 1-> one time error in division followed by relaxation to mean, if = 0 -> random error at every division
        self.single_division_error_regulator_lipids = 0 * np.ones(
            self.n_series)  # either 1 or 0, 1-> one time error in division followed by relaxation to mean, if = 0 -> random error at every division
        self.noise_strength_lipids = 5000 # magnitude of the noise in the lipid concentraiton [default: 5000]
        self.relaxation_rate = self.rate_growth  # timescale at which lipid noise relaxes back to mean, [default: self.rate_growth}, if larger, then processes like active degradation and regulation of production rate matter
        self.lipid_conc_0 = 1 * self.activation_rate_lipids/self.rate_growth # lipid concentration at beginning of simulation [default: activation_rate_lipids/rate_growth], in figure 4 of paper initial concentration is 1.3 times default
        self.single_lipid_conc_perturb = 0 # either 0 or 1, if 1 -> single perturbation of lipid concentration and the deterministic relaxation, if 0-> lipid noise in concentration at every time step [default: 0]
        self.time_of_perturb = 0 # time of the single perturbation in units of hours [default: 0]

        # Parameters for modelling RIDA explicitly
        self.model_rida_explicitly = 0 # either 0 or 1, if 0 -> rida deactivation rate is constant, if 1-> fluctuations in rida deactivation rate at every time step
        self.noise_strength_rida = 100 # magnitude of the noise in the rida concentraiton [default: 100]

        # Parameters for introducing division and partitionning noise
        self.cv_division_position = 0 * np.ones(
            self.n_series)  # coefficient of variation of division position, if 1 -> two daughter cells have different volumes after division, if 0-> no division noise [default: 0]
        self.partitionning_error_initiator = 0 * np.ones(
            self.n_series)  # either 0 or 1, if 1 -> unequal paritionning of the initiator proteins in the two cells, if 0 ->  number of initiators is divided by two at division [default: 0]
        self.partitionning_error_regulator = 0 * np.ones(
            self.n_series)  # either 0 or 1, if 1 -> unequal paritionning of the regulator proteins in the two cells, if 0 ->  number of regulators is divided by two at division [default: 0]
        self.single_division_error = 0 * np.ones(
            self.n_series)  # if ==1, one time error in division followed by relaxation to mean, if == 0, random error at every division [default: 0]
        self.cycle_with_error = 10  # cycle at which single division error occurs [default: 10]
        self.single_protein_number_error = 0  # if ==1, one time error in protein partitioning followed by relaxation to mean, if == 0, random error at every division [default: 0]
        self.prod_rate_prop_to_v = 1 * np.ones(
            self.n_series)  # proportionality factor between the volume and the production rate of proteins, [default: 1], typically not varied
        
        # Parameters for the growing cell model of gene expression
        self.finite_dna_replication_rate = 1 # either 1 or 0, if 1 -> effect of finite time to replicate chromosome on gene allocation fraction is included
        self.number_density = 10**6 * np.ones(self.n_series) # number density in growing cell model [default: 10**6]
        if self.gene_expression_model == 'ribo_limiting': # redefine new basal rate 0 and set basal rate to basal_rate_0 times the growth rate, as follows from growing cell model
            self.basal_rate_initiator_0 =  self.basal_rate_initiator
            self.basal_rate_regulator_0 = self.basal_rate_regulator
            self.basal_rate_initiator = self.basal_rate_initiator_0 * self.rate_growth
            self.basal_rate_regulator = self.basal_rate_regulator_0 * self.rate_growth
            self.gene_alloc_fraction_initiator_0 = self.basal_rate_initiator_0 / self.number_density
            self.gene_alloc_fraction_regulator_0 = self.basal_rate_regulator_0 / self.number_density

        # Parameters for modifications of the gene expression of DnaA
        self.stochastic_gene_expression = 0 # either 0 or 1, if 1-> gene expression of DnaA is stochastic [default: 0]
        self.noise_strength_total_dnaA = 100 * np.ones(self.n_series) # magnitude of the noise in the rida concentraiton [default: 100]
        self.block_production = 1 # either 0 or 1, if 1 -> DnaA production is stopped during the blocked period after replication initaition [default: 0]
        self.external_oscillations = 0 # either 0 or 1, if 1-> include externally driven osciallations in the DnaA gene expression as in experiments done by Si et al. 2019 [default: 0]
        self.amplitude_oscillations = 30 * np.ones(self.n_series) # amplitude of the externally driven oscillations [default: 30]
        self.period_oscillations = 2 / self.doubling_rate * np.ones(self.n_series)
        self.offset_oscillations = 350 * np.ones(self.n_series) #self.amplitude_oscillations #self.total_conc * 0.9 #
        self.continuous_oscillations = 1 # either 0 or 1, if 1-> oscillations follow sine function, if 0 -> oscillations are following step function
        self.underexpression_oscillations = 0 # either 0 or 1, if 1-> DnaA is underexpressed and native DnaA expression is removed, if 0 -> both external oscillations and native expression are adding to total DnaA expression

        # Parameters for different versions of regulating cell division 
        self.independent_division_cycle = 0 * np.ones(self.n_series) # if 1 -> division cycle is not coupled to replication cycle, if 0 -> replication initiation triggers cell division [default: 0]
        self.version_of_independent_division_regulation = 'IDA'  # if division is regulated independently of replication, there are different version to trigger cell division [default: 'IDA']:
                                                        # 'IDA' -> independent double adder, independently added volume from birth to division
                                                        # 'sizer' -> constant division volume independent of replication or birth volume
        self.version_of_coupled_division_regulation = 'cooper'  # if division is triggered by replication initiation, these are the differnt coupling mechanisms: [default: 'cooper']
                                                        # 'cooper' -> cell divides constant tcc after replication initiation
                                                        # 'RDA' -> replication double adder, independently added volume from replication initiation to division
        self.division_volume = 0.4  * np.ones(self.n_series) # if replication is controlled independently, then this is the critical added/division size
        self.v_init_th = 0.28 * np.ones(self.n_series) # the theoretical volume at which replication is initiated as reported by experiments [default:0.28]
        self.cv_added_volume = 0.1 # coefficient of variation of the volume added [default: 0.1]

    @property
    def parameter_set_df(self):
        """ Returns a data frame of the instance of the ParameterSet class."""
        parameter_set_dict = vars(self)
        return pd.DataFrame.from_dict(parameter_set_dict)

    def calculate_dars2_from_rida_rate(self):
        return self.deactivation_rate_rida * self.init_conc / (self.michaelis_const_destr + self.init_conc) \
               * (self.michaelis_const_prod + self.total_conc - self.init_conc) / (self.total_conc - self.init_conc)

    def calculate_v_init_from_parameter_set(self):
        if self.model_lipids_explicitly ==1:
            activation_rate_lipids = self.activation_rate_lipids / self.rate_growth
        else:
            activation_rate_lipids = self.activation_rate_lipids
        if self.version_of_model == 'switch_critical_frac':
            return (self.deactivation_rate_rida + self.deactivation_rate_datA) / \
                   activation_rate_lipids * self.frac_init * \
                   (self.michaelis_const_prod + (1 - self.frac_init) * self.total_conc) / \
                   ((1 - self.frac_init) * (
                           self.michaelis_const_destr + self.frac_init *
                           self.total_conc)) - \
                   (self.activation_rate_dars2 + self.activation_rate_dars1) / activation_rate_lipids
        if self.version_of_model == 'switch':
            return (self.deactivation_rate_rida + self.deactivation_rate_datA) / \
                   activation_rate_lipids * self.init_conc * \
                   (self.michaelis_const_prod + self.total_conc - self.init_conc) / \
                   ((self.total_conc - self.init_conc) * (self.michaelis_const_destr + self.init_conc)) - \
                   (self.activation_rate_dars2 + self.activation_rate_dars1) / activation_rate_lipids
        if self.version_of_model == 'titration':
            return self.n_c_max_0 /(self.michaelis_const_initiator - self.critical_free_conc)