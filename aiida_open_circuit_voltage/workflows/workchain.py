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
from aiida_quantumespresso.common.types import RelaxType, SpinType
from aiida_open_circuit_voltage.calculations.functions import functions as func 

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
        spec.expose_inputs(PwRelaxWorkChain, namespace='ocv_relax',
            exclude=('clean_workdir', 'structure'),
            namespace_options={'help': 'Inputs for the `PwRelaxWorkChain` for running the four relax calculations are called in the `ocv_relax` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input unitcell structure.')
        spec.input('ocv_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing the parameters used to calculate OCVs.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup,
            cls.run_relax_discharged,
            cls.run_relax_charged,
            cls.build_supercells,
            cls.run_relax_low_SOC,
            cls.run_relax_high_SOC,
            cls.results,
        )

        spec.exit_code(202, 'ERROR_STRUCTURE_NOT_FOUND',
            message='The output relaxed structure of PwRelaxWorkChains not found.')
        spec.exit_code(203, 'ERROR_DFT_ENERGY_NOT_FOUND',
            message='The energy from the final scf calculation not found.')
        spec.output('open_circuit_voltages', valid_type=orm.Dict,
            help='The dictionary containing the three voltages - average ocv and the ocv at high and low SOCs.')
        
    def setup(self):
        """Input validation and context setup."""

        self.ctx.discharged_unitcell = self.inputs.structure

        # I store input dictionaries in context variables
        self.ctx.ocv_parameters_d = self.inputs.ocv_parameters.get_dict()

        self.ctx.ocv_relax = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='ocv_relax'))
        self.ctx.ocv_relax.base.pw.parameters = self.ctx.ocv_relax.base.pw.parameters.get_dict()
        self.ctx.ocv_relax.base.pw.settings = self.ctx.ocv_relax.base.pw.settings.get_dict()
        self.ctx.ocv_relax.base_final_scf.pw.parameters = self.ctx.ocv_relax.base_final_scf.pw.parameters.get_dict()
        self.ctx.ocv_relax.base_final_scf.pw.settings = self.ctx.ocv_relax.base_final_scf.pw.settings.get_dict()


    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files
        from aiida_open_circuit_voltage.workflows import protocols as proto
        return files(proto) / 'ocv.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, protocol=None, overrides=None, **kwargs
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

        args = (code, structure, protocol)
        ocv_relax = PwRelaxWorkChain.get_builder_from_protocol(*args, overrides=inputs['ocv_relax'], **kwargs)

        ocv_relax.pop('structure', None)
        ocv_relax.pop('clean_workdir', None)
        ocv_relax.pop('parent_folder', None)

        builder = cls.get_builder()
        builder.ocv_relax = ocv_relax

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        builder.ocv_parameters = orm.Dict(dict=inputs['ocv_parameters'])

        return builder

    def run_relax_discharged(self):
        """
        Runs a PwRelaxWorkChain to relax the input (discharged) i.e. completely lithiated structure.
        """
        inputs = self.ctx.ocv_relax

        self.ctx.discharged_unitcell.set_extra('relaxed', False)
        self.ctx.discharged_unitcell.set_extra('supercell', False)

        inputs['structure'] = self.ctx.discharged_unitcell
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'discharged_relax'
        inputs.metadata.label = 'discharged_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}> on discharged structure')
        
        return ToContext(relax_workchains=append_(running))

    def run_relax_charged(self):
        """
        Runs a PwRelaxWorkChain to relax the charged i.e. completely delithiated structure.
        """

        # Saving the relaxed structure in context variable
        workchain = self.ctx.relax_workchains[-1]
        self.ctx.discharged_unitcell_relaxed = workchain.outputs.output_structure
        self.ctx.charged_unitcell_rel.set_extra('relaxed', True)
        self.ctx.charged_unitcell_rel.set_extra('supercell', False)

        inputs = self.ctx.ocv_relax

        self.ctx.charged_unitcell = func.get_charged(self.ctx.discharged_unitcell)

        self.ctx.charged_unitcell.set_extra('relaxed', False)
        self.ctx.charged_unitcell.set_extra('supercell', False)

        inputs['structure'] = self.ctx.charged_unitcell
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'charged_relax'
        inputs.metadata.label = 'charged_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}> on charged structure')
        
        return ToContext(relax_workchains=append_(running))

    def build_supercells(self):
        """ Making all types of supercells here. 
        """
        # Saving the relaxed structures in context variables
        workchain_discharged = self.ctx.relax_workchains[0]
        workchain_charged = self.ctx.relax_workchains[1]

        try: 
            self.ctx.discharged_unitcell_relaxed = workchain_discharged.outputs.output_structure
            self.ctx.charged_unitcell_relaxed = workchain_charged.outputs.output_structure
        except exceptions.NotExistent:
            self.report('the PwRelaxWorkChains did not generate output structures')
            return self.exit_codes.ERROR_STRUCTURE_NOT_FOUND

        self.ctx.discharged_unitcell_relaxed.set_extra('relaxed', True)
        self.ctx.discharged_unitcell_relaxed.set_extra('supercell', False)
        self.ctx.charged_unitcell_relaxed.set_extra('relaxed', True)
        self.ctx.charged_unitcell_relaxed.set_extra('supercell', False)

        distance = self.ctx.ocv_parameters_d['distance']

        self.ctx.charged_supercell_relaxed = func.make_supercell(self.ctx.charged_unitcell_relaxed, orm.Float(distance))
        self.ctx.discharged_supercell_relaxed = func.make_supercell(self.ctx.discharged_unitcell_relaxed, orm.Float(distance))

        # To do - suggest changes to distance such that this assertion doesn't fail
        assert self.ctx.charged_unitcell.extras['missing_cations'] * len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell.sites) + len(self.ctx.charged_supercell_relaxed.sites) == len(self.ctx.discharged_supercell_relaxed.sites), 'Change distance in make_supercell()'

        self.ctx.charged_supercell_relaxed.set_extra('relaxed', True)
        self.ctx.charged_supercell_relaxed.set_extra('supercell', True)

        self.ctx.discharged_supercell_relaxed.set_extra('relaxed', True)
        self.ctx.discharged_supercell_relaxed.set_extra('supercell', True)

        all_cation_indices, unique_cation_indices = func.get_unique_cation_sites(self.ctx.discharged_supercell_relaxed)
        self.ctx.low_SOC_supercells = func.get_low_SOC(self.ctx.discharged_supercell_relaxed, unique_cation_indices) 
        self.ctx.high_SOC_supercells = func.get_high_SOC(self.ctx.discharged_supercell_relaxed, self.ctx.charged_supercell_relaxed, all_cation_indices, unique_cation_indices) 

        return

    def run_relax_low_SOC(self):
        """
        Runs a PwRelaxWorkChain to relax the supercells with one cation removed.
        """
        inputs = self.ctx.ocv_relax

        if self.ctx.ocv_parameters_d['SOC_supercells'] == 'random':
            inputs['structure'] = self.ctx.low_SOC_supercells[0]
        
        elif self.ctx.ocv_parameters_d['SOC_supercells'] == 'all':
            raise NotImplementedError('Relaxing all low SOC supercells is not implemented yet')

        if self.ctx.ocv_parameters_d['SOC_relax'] == 'fixed_cell':
            inputs['relax_type'] = RelaxType.POSITIONS
        
        elif self.ctx.ocv_parameters_d['SOC_relax'] == 'variable_cell':
            inputs['relax_type'] = RelaxType.POSITIONS_CELL
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'low_SOC_relax'
        inputs.metadata.label = 'low_SOC_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}>')
        
        return ToContext(relax_workchains=append_(running))

    def run_relax_high_SOC(self):
        """
        Runs a PwRelaxWorkChain to relax the supercells with only one cation remaining.
        """
        inputs = self.ctx.ocv_relax

        if self.ctx.ocv_parameters_d['SOC_supercells'] == 'random':
            inputs['structure'] = self.ctx.high_SOC_supercells[0]
        
        elif self.ctx.ocv_parameters_d['SOC_supercells'] == 'all':
            raise NotImplementedError('Relaxing all low SOC supercells is not implemented yet')

        if self.ctx.ocv_parameters_d['SOC_relax'] == 'fixed_cell':
            inputs['relax_type'] = RelaxType.POSITIONS
        
        elif self.ctx.ocv_parameters_d['SOC_relax'] == 'variable_cell':
            inputs['relax_type'] = RelaxType.POSITIONS_CELL
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'high_SOC_relax'
        inputs.metadata.label = 'high_SOC_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}>')
        
        return ToContext(relax_workchains=append_(running))
    def results(self):
        """Retrun the output trajectory and diffusion coefficients generated in the last MD run."""
        if self.ctx.converged and self.ctx.diffusion_counter <= self.ctx.ocv_parameters_d['max_ld_iterations']:
            self.report(f'workchain completed after {self.ctx.diffusion_counter} iterations')
        else:
            self.report('maximum number of LinDiffusion convergence iterations exceeded')

        self.out('msd_results', self.ctx.relax_workchains[-1].outputs.msd_results)
        self.out('total_trajectory', self.ctx.relax_workchains[-1].outputs.total_trajectory)
        self.out('coefficients', self.ctx.workchains_fitting[-1].outputs.coefficients)
