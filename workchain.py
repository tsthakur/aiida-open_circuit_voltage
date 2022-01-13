# -*- coding: utf-8 -*-
"""Mother Workchain that calls PwRelaxWorkChain and PwBaseWorkChain to relax experimental structures and
calculate DFT energies used to compute open circuit voltages (OCV) at low and high state of charge (SOC) 
and average OCV for any arbitrary cathode material"""
import numpy as np
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.engine import ToContext, append_, if_, while_, WorkChain
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.plugins import WorkflowFactory

PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

class OCVWorkChain(ProtocolMixin, WorkChain): # maybe BaseRestartWorkChain?
    """The main (and only?) Workchain of aiida-open_circuit_voltage that calls aiida-quantumespresso 
    workchains to run PwRelaxWorkChains to calculate OCV at various SOCs using Quantum ESPRESSO pw.x.

    The main workflow is as follows - 

    vc-relax user provided completely discharged unitcell strcuture with pwRelaxWorkChain
    remove all cations from unitcell and vc-relax this completely charged unitcell with pwRelaxWorkChain
    estimate VOC_avg with these 3 pieces of informations

    build charged and discharged supercells

    We assume removing 1 cation doesn't distort the discharged supercell and similarly adding 1 cation 
    doesn't distort the charged supercell

    Low SOC
    make list of supercells after removing 1 (symmetery inequivalent) cation
    based on user input either do fixed cell relax of one such supercell or do fixed cell relax of all unique supercells and take the lowest formation energy

    High SOC
    scale the discharged supercell with the lattice vectors of charged supercell and remove all but 1 cation
    make list of sypercells after leaving 1 (symmetery inequivalent) cation
    based on user input either do fixed cell relax of one such supercell or do fixed cell relax of all unique supercells and take the lowest formation energy
    """
    # _process_class = PwRelaxWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax',
            exclude=('clean_workdir', 'structure'),
            namespace_options={'help': 'Inputs for the `PwRelaxWorkChain` for estimating diffusion coefficient with MD runs are called in the `ld` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input unitcell structure.')
        spec.input('ocv_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing the parameters used to calculate OCVs.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
            cls.run_lindiff,
            cls.run_fit,
            cls.inspect_process,
            ),
            cls.run_last_lindiff,
            cls.results,
        )

        spec.exit_code(707, 'ERROR_FITTING_FAILED',
            message='The linear regression to fit pinball and dft forces failed.')
        spec.exit_code(703, 'ERROR_CHARGE_DENSITIES_NOT_FOUND',
            message='Either the stashed charge densities or the flipper compatible supercell structure not found.')
        spec.exit_code(708, 'ERROR_LINDIFFUSION_FAILED',
            message='The LinDiffusionWorkChain sub process failed.')
        spec.exit_code(704, 'ERROR_TRAJECTORY_NOT_FOUND',
            message='The output trajectory of ReplayMDWorkChain not found.')
        # spec.expose_outputs(ReplayMDWorkChain)
        spec.output('msd_results', valid_type=orm.ArrayData,
            help='The dictionary containing the results of msd calculations using the samos library of the last LinDiffusionWorkChain')
        spec.output('total_trajectory', valid_type=orm.TrajectoryData,
            help='The full concatenated trajectory of the last LinDiffusinWorkChains.')
        spec.output('coefficients', valid_type=orm.Dict,
            help='The dictionary containing the final pinball hyperparameters(keyword - `coefs`) along with linear regression values.')
        
    def setup(self):
        """Input validation and context setup."""

        self.ctx.diffusion_counter = 0
        self.ctx.converged = False
        self.ctx.current_structure = self.inputs.structure

        # I store input dictionaries in context variables
        self.ctx.diffusion_convergence_parameters_d = self.inputs.diffusion_convergence_parameters.get_dict()

        self.ctx.lindiff_inputs = AttributeDict(self.exposed_inputs(LinDiffusionWorkChain, namespace='ld'))
        self.ctx.fitting_inputs = AttributeDict(self.exposed_inputs(FittingWorkChain, namespace='ft'))
        self.ctx.lindiff_inputs.md.pw.settings = self.ctx.lindiff_inputs.md.pw.settings.get_dict()
        # Without putting as a dict inside the namespace of lindiff, msd_parameters can't be updated later
        self.ctx.lindiff_inputs.msd_parameters = self.ctx.lindiff_inputs.msd_parameters.get_dict()
        self.ctx.lindiff_inputs.diffusion_parameters = self.ctx.lindiff_inputs.diffusion_parameters.get_dict()
        # I store this in context variable to update for every MD run after the first one
        self.ctx.max_lindiff_iterations = self.ctx.lindiff_inputs.diffusion_parameters['max_md_iterations']
        if self.ctx.lindiff_inputs.get('coefficients'):
            self.report(f'I was given pinball coefficients <{self.ctx.lindiff_inputs.coefficients.pk}>')
            self.ctx.diffusion_counter = 1
            # Adding the fitting and lindiff workchains that generated this pinball parameter 
            qb = orm.QueryBuilder()
            qb.append(orm.Dict, filters={'uuid':{'==':self.ctx.lindiff_inputs.coefficients.uuid}}, tag='coefs')
            qb.append(WorkflowFactory('quantumespresso.flipper.fitting'), with_outgoing='coefs', tag='fit')
            self.ctx.workchains_fitting = qb.all(flat=True)
            qb.append(orm.TrajectoryData, with_outgoing='fit', tag='traj')
            qb.append(WorkflowFactory('quantumespresso.flipper.lindiffusion'), with_outgoing='traj', tag='lindiff')
            self.ctx.workchains_lindiff = qb.all(flat=True)

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_flipper.workflows import protocols as proto
        return files(proto) / 'condiff.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, parent_folder, protocol=None, overrides=None, **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol, usually takes the pseudo potential family.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (code, structure, parent_folder, protocol)
        lindiff = LinDiffusionWorkChain.get_builder_from_protocol(*args, overrides=inputs['ld'], **kwargs)

        lindiff.pop('structure', None)
        lindiff.pop('clean_workdir', None)
        lindiff.pop('parent_folder', None)

        args = (code, structure, parent_folder, protocol)
        fitting = FittingWorkChain.get_builder_from_protocol(*args, overrides=inputs['ft'], **kwargs)

        fitting.pop('structure', None)
        fitting.pop('clean_workdir', None)
        fitting.pop('parent_folder', None)

        builder = cls.get_builder()
        builder.ld = lindiff
        builder.ft = fitting

        builder.structure = structure
        builder.parent_folder = parent_folder
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.diffusion_convergence_parameters = orm.Dict(dict=inputs['diffusion_convergence_parameters'])

        return builder

    def should_run_process(self):
        """Return whether a lindiffusion and fitting workchains should be run.

        This is the case as long as the last process has not finished successfully, and the number of maximum replays has not been reached or diffusion coefficient converged or minimum number of replays has not been reached.
        """
        if (self.ctx.diffusion_convergence_parameters_d['min_ld_iterations'] > self.ctx.diffusion_counter): return True
        elif (not(self.ctx.converged) and (self.ctx.diffusion_convergence_parameters_d['max_ld_iterations'] > self.ctx.diffusion_counter)): return True
        else: return False

    def run_lindiff(self):
        """
        Runs a LinDiffusionWorkChain for an estimate of the diffusion.
        If there is a last fitting estimate, I update the parameters for the pinball.
        """
        inputs = self.ctx.lindiff_inputs
        inputs['parent_folder'] = self.inputs.parent_folder

        if (self.ctx.diffusion_counter == 0):
            # if this is first run, then I launch an unmodified LinDiffusionWorkChain
            inputs['structure'] = self.ctx.current_structure
            # Since this is a first run, I don't want to run for too long
            inputs.diffusion_parameters.update({'max_md_iterations': 1})

        else:
            # for every run after the first one, the pinball hyperparameters are taken from the output of the
            # last fitting workchain, which used the output trajectory of previous MD run to do fitting

            workchain = self.ctx.workchains_lindiff[-1]
            create_missing = len(self.ctx.current_structure.sites) != workchain.outputs.total_trajectory.get_attribute('array|positions')[1]
            # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
            kwargs = dict(trajectory=workchain.outputs.total_trajectory, 
                        parameters=get_or_create_input_node(orm.Dict, dict(
                            step_index=-1,
                            recenter=False,
                            create_settings=True,
                            complete_missing=create_missing), store=False),)
            if create_missing:
                kwargs['structure'] = self.ctx.current_structure
                kwargs['settings'] = get_or_create_input_node(orm.Dict, self.ctx.lindiff_inputs.md.pw.settings, store=False)

            res = get_structure_from_trajectory(**kwargs)
            
            inputs['structure'] = res['structure']
            self.ctx.current_structure = res['structure']
            inputs.md['pw']['parameters']['IONS'].update({'ion_velocities': 'from_input'})
            inputs.md['pw']['settings'] = res['settings'].get_dict()
            # Since I start from previous trajectory, it has sufficiently equilibriated 
            inputs.msd_parameters.update({'equilibration_time_fs': 0})
            # I need to use the input value for every run after first one
            inputs.diffusion_parameters.update({'max_md_iterations': self.ctx.max_lindiff_iterations})
            # Updating the pinball hyperparameters
            inputs.coefficients = self.ctx.workchains_fitting[-1].outputs.coefficients
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'
        inputs.metadata.label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'

        inputs = prepare_process_inputs(LinDiffusionWorkChain, inputs)
        running = self.submit(LinDiffusionWorkChain, **inputs)

        self.report(f'launching LinDiffusionWorkChain<{running.pk}>')
        
        return ToContext(workchains_lindiff=append_(running))

    def run_fit(self):
        """
        Runs a fitting workflow on positions taken from the output of the previous lindiff run
        """
        inputs = self.ctx.fitting_inputs
        inputs['parent_folder'] = self.inputs.parent_folder
        # There's no difference between the first and subsequent runs so I don't change anything
        inputs['structure'] = self.ctx.current_structure
        inputs.md['hustler_snapshots'] = self.ctx.workchains_lindiff[-1].outputs.total_trajectory
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'fitting_{self.ctx.diffusion_counter:02d}'
        inputs.metadata.label = f'fitting_{self.ctx.diffusion_counter:02d}'

        inputs = prepare_process_inputs(FittingWorkChain, inputs)
        running = self.submit(FittingWorkChain, **inputs)

        self.ctx.diffusion_counter +=1
        self.report(f'launching FittingWorkChain<{running.pk}>')
        
        return ToContext(workchains_fitting=append_(running))

    def inspect_process(self):
        """Inspect the results of the last `LinDiffusionWorkChain` and `FittingWorkChain`.

        I compute the MSD from the previous trajectory and check if it converged with respect to the provided threshold, both relative and absolute.
        """
        # I don't need to check much, since the daughter workchains take care of themselves
        try:
            species = self.ctx.lindiff_inputs.msd_parameters['species_of_interest'][0]
        except: 
            species = 'Li'
        param_d = self.ctx.diffusion_convergence_parameters_d
        # I will start checking when minimum no. of iterations are reached
        if self.ctx.diffusion_counter >= param_d['min_ld_iterations']:
            # Since I am here, it means I need to check the last 3 calculations to
            # see whether I converged or need to run again:
            # Now let me see the pinball coefficients that I get and if they have converged
            # I consider it converged if either the last 3 estimates have not changed more than the threshold or the difference of the last 2 estimates is within the threshold

            coefficients = np.array([workchain_fitting.outputs.coefficients.get_dict()['coefs'] for workchain_fitting in self.ctx.workchains_fitting])
            stddev_3 = np.std(coefficients[-3:], axis=0)
            # checking the variation (relative error) in last 2 fits, if the standard deviation of last 3 fits is too high
            difference_2 = abs((coefficients[-1]-coefficients[-2])/coefficients[-1])

            if (stddev_3 < param_d['coefficient_threshold_std']).all() and (difference_2 < param_d['coefficient_threshold_diff']).all():
                # I have converged, yay me!
                self.report(f'Diffusion converged with std = {stddev_3} < threshold = {param_d["coefficient_threshold_std"]}')
                self.ctx.converged = True
            elif (difference_2 < param_d['coefficient_threshold_diff']).all():
                self.report(f'Last two estimates of Pinball parameters have converged with relative error = {difference_2} < threshold = {param_d["coefficient_threshold_diff"]}')
                self.ctx.converged = True
            elif (stddev_3 < param_d['coefficient_threshold_std']).any() and (difference_2 < param_d['coefficient_threshold_diff']).any():
                self.report(f'Not all Pinball parameters have converged with std = {stddev_3} and relative error = {difference_2}, so I start another MD iteration')
                self.ctx.converged = False
            else:
                self.report(f'The Pinball parameters have not converged with std = {stddev_3} and relative error = {difference_2}')
                self.ctx.converged = False

        return

    def run_last_lindiff(self):
        """
        Runs a final LinDiffusionWorkChain after converging Pinball parameters, starting from the previous trajectory.
        This is the MD run that to be used for all post processing.
        """
        inputs = self.ctx.lindiff_inputs
        inputs['parent_folder'] = self.inputs.parent_folder

        # I use the last estimated Pinball parameters along with the last output trajectory
        workchain = self.ctx.workchains_lindiff[-1]
        create_missing = len(self.ctx.current_structure.sites) != workchain.outputs.total_trajectory.get_attribute('array|positions')[1]
        # create missing tells inline function to append additional sites from the structure that needs to be passed in such case
        kwargs = dict(trajectory=workchain.outputs.total_trajectory, 
                    parameters=get_or_create_input_node(orm.Dict, dict(
                        step_index=-1,
                        recenter=False,
                        create_settings=True,
                        complete_missing=create_missing), store=False),)
        if create_missing:
            kwargs['structure'] = self.ctx.current_structure
            kwargs['settings'] = get_or_create_input_node(orm.Dict, self.ctx.lindiff_inputs.md.pw.settings, store=False)

        res = get_structure_from_trajectory(**kwargs)
        
        inputs['structure'] = res['structure']
        self.ctx.current_structure = res['structure']
        inputs.md['pw']['parameters']['IONS'].update({'ion_velocities': 'from_input'})
        inputs.md['pw']['settings'] = res['settings'].get_dict()
        # Since I start from previous trajectory, it has sufficiently equilibriated 

        # Updating the pinball hyperparameters
        inputs.coefficients = self.ctx.workchains_fitting[-1].outputs.coefficients

        # Starting from previous trajectory
        inputs.md.previous_trajectory = workchain.outputs.total_trajectory
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'
        inputs.metadata.label = f'lindiffusion_{self.ctx.diffusion_counter:02d}'

        inputs = prepare_process_inputs(LinDiffusionWorkChain, inputs)
        running = self.submit(LinDiffusionWorkChain, **inputs)

        self.report(f'launching LinDiffusionWorkChain<{running.pk}>')
        
        return ToContext(workchains_lindiff=append_(running))

    def results(self):
        """Retrun the output trajectory and diffusion coefficients generated in the last MD run."""
        if self.ctx.converged and self.ctx.diffusion_counter <= self.ctx.diffusion_convergence_parameters_d['max_ld_iterations']:
            self.report(f'workchain completed after {self.ctx.diffusion_counter} iterations')
        else:
            self.report('maximum number of LinDiffusion convergence iterations exceeded')

        self.out('msd_results', self.ctx.workchains_lindiff[-1].outputs.msd_results)
        self.out('total_trajectory', self.ctx.workchains_lindiff[-1].outputs.total_trajectory)
        self.out('coefficients', self.ctx.workchains_fitting[-1].outputs.coefficients)