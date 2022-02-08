# -*- coding: utf-8 -*-
"""
Mother Workchain that calls PwRelaxWorkChain to relax experimental structures and
calculate DFT energies used to compute open circuit voltages (OCV) at low and high state of 
charge (SOC) and average OCV for any arbitrary cathode material
"""
import struct
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
    """
    The main (and only?) Workchain of aiida-open_circuit_voltage that calls aiida-quantumespresso 
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
        """
        Define the process specification.
        """
        super().define(spec)
        spec.expose_inputs(PwRelaxWorkChain, namespace='ocv_relax',
            exclude=('clean_workdir', 'structure'),
            namespace_options={'help': 'Inputs for the `PwRelaxWorkChain` for running the four relax calculations are called in the `ocv_relax` namespace.'})
        spec.input('structure', valid_type=orm.StructureData, help='The input unitcell structure.')
        spec.input('ocv_parameters', valid_type=orm.Dict, required=False, help='The dictionary containing the parameters used to calculate OCVs.')
        spec.input('discharged_unitcell_relaxed', valid_type=orm.StructureData, required=False, help='The relaxed unitcell needed to restart this workchain.')
        spec.input('charged_unitcell_relaxed', valid_type=orm.StructureData, required=False, help='The relaxed unitcell needed to restart this workchain.')
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
        """
        Input validation and context setup.
        """
        self.ctx.discharged_unitcell = self.inputs.structure

        # I store input dictionaries in context variables
        self.ctx.ocv_parameters_d = self.inputs.ocv_parameters.get_dict()
        cation = orm.Str(self.ctx.ocv_parameters_d['cation'])
        self.ctx.cation = cation

        # setting up none values to check in future if they have been inputted or generated by PwRelaxWorkChains 
        self.ctx.discharged_unitcell_relaxed = None
        self.ctx.charged_unitcell_relaxed = None

        self.ctx.ocv_relax = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='ocv_relax'))
        self.ctx.ocv_relax.base.pw.parameters = self.ctx.ocv_relax.base.pw.parameters.get_dict()
        self.ctx.ocv_relax.base_final_scf.pw.parameters = self.ctx.ocv_relax.base_final_scf.pw.parameters.get_dict()

    @classmethod
    def get_protocol_filepath(cls):
        """
        Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols.
        """
        from importlib_resources import files
        from aiida_open_circuit_voltage.workflows import protocols as proto
        return files(proto) / 'ocv.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls, code, structure, discharged_unitcell_relaxed=None, charged_unitcell_relaxed=None, protocol=None, overrides=None, **kwargs
    ):
        """
        Return a builder prepopulated with inputs selected according to the chosen protocol.

        !!Note: If providing the relaxed unitcells, provide their single scf energies as well in the ocv_parameters dictionary!!

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param discharged_unitcell_relaxed: the ``StructureData`` instance that has been already relaxed.
        :param charged_unitcell_relaxed: the ``StructureData`` instance that has all the cations removed and has been relaxed.
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
        if discharged_unitcell_relaxed: builder.discharged_unitcell_relaxed = discharged_unitcell_relaxed
        if charged_unitcell_relaxed: builder.charged_unitcell_relaxed = charged_unitcell_relaxed

        if inputs['ocv_parameters']['cation'] not in ['Li', 'Mg']: 
            raise NotImplemented('Only Li and Mg ion materials supported now.')

        return builder

    def run_relax_discharged(self):
        """
        Runs a PwRelaxWorkChain to relax the input (discharged) i.e. completely ionised structure.
        """
        if self.inputs.get('discharged_unitcell_relaxed'):
            self.report(f'Relaxed discharged unitcell <{self.inputs.discharged_unitcell_relaxed.pk}> already provided, skipping to next step...')
            self.ctx.discharged_unitcell_relaxed = self.inputs.discharged_unitcell_relaxed
            return
            
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
        Runs a PwRelaxWorkChain to relax the charged i.e. completely deionised structure.
        To do - run this relax chain in parallel with previous one.
        """
        # Saving the relaxed structure in context variable
        if not self.ctx.discharged_unitcell_relaxed:
            try:
                self.ctx.discharged_unitcell_relaxed = self.ctx.relax_workchains[-1].outputs.output_structure
            except exceptions.NotExistent:
                self.report('The PwRelaxWorkChains did not generate output structures')
                return self.exit_codes.ERROR_STRUCTURE_NOT_FOUND        
                
        if self.inputs.get('charged_unitcell_relaxed'):
            self.report(f'Relaxed charged unitcell <{self.inputs.charged_unitcell_relaxed.pk}> already provided, skipping to next step...')
            self.ctx.charged_unitcell_relaxed = self.inputs.charged_unitcell_relaxed
            return

        inputs = self.ctx.ocv_relax

        self.ctx.charged_unitcell = func.get_charged(self.ctx.discharged_unitcell, self.ctx.cation)['decationised_structure']

        self.ctx.charged_unitcell.set_extra('relaxed', False)
        self.ctx.charged_unitcell.set_extra('supercell', False)

        inputs['structure'] = self.ctx.charged_unitcell

        # Removing Li pseudopotential since this structure no longer has Li in it
        self.ctx.Li_pseudo = inputs['base']['pw']['pseudos'].pop('Li')
        inputs['base_final_scf']['pw']['pseudos'].pop('Li')

        # Adding missing charge due to Li ions
        inputs.base.pw.parameters['SYSTEM']['tot_charge'] = float(-self.ctx.charged_unitcell.extras['missing_cations'])
        inputs.base_final_scf.pw.parameters['SYSTEM']['tot_charge'] = float(-self.ctx.charged_unitcell.extras['missing_cations'])
        
        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'charged_relax'
        inputs.metadata.label = 'charged_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> on charged structure')
        return ToContext(relax_workchains=append_(running))

    def build_supercells(self):
        """ 
        Making all types of supercells here. 
        """
        # Saving the relaxed structures in context variables
        if not self.ctx.charged_unitcell_relaxed:
            try:
                self.ctx.charged_unitcell_relaxed = self.ctx.relax_workchains[-1].outputs.output_structure
            except exceptions.NotExistent:
                self.report('The PwRelaxWorkChains did not generate output structures')
                return self.exit_codes.ERROR_STRUCTURE_NOT_FOUND

        self.ctx.discharged_unitcell_relaxed.set_extra('relaxed', True)
        self.ctx.discharged_unitcell_relaxed.set_extra('supercell', False)
        self.ctx.charged_unitcell_relaxed.set_extra('relaxed', True)
        self.ctx.charged_unitcell_relaxed.set_extra('supercell', False)

        distance = self.ctx.ocv_parameters_d['distance']
        distance_upperbound = self.ctx.ocv_parameters_d['distance_upperbound']
        distance_epsilon = self.ctx.ocv_parameters_d['distance_epsilon']

        self.ctx.charged_supercell_relaxed = func.make_supercell(self.ctx.charged_unitcell_relaxed, distance)
        self.ctx.discharged_supercell_relaxed = func.make_supercell(self.ctx.discharged_unitcell_relaxed, distance)

        # Checking if the two supercells have same no. of non cationic atoms and making different supercells if not
        if len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell_relaxed.sites) == len(self.ctx.discharged_supercell_relaxed.sites) / len(self.ctx.discharged_unitcell_relaxed.sites):
            self.report(f'Distance<{distance}> given to build supercells was adequate, so continuing with it')

        elif len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell_relaxed.sites) > len(self.ctx.discharged_supercell_relaxed.sites) / len(self.ctx.discharged_unitcell_relaxed.sites):
            # Keep making new supercells till it their sizes match
            for dist in np.arange(distance, distance_upperbound, distance_epsilon):
                self.ctx.charged_supercell_relaxed = func.make_supercell(self.ctx.charged_unitcell_relaxed, distance)
                self.ctx.discharged_supercell_relaxed = func.make_supercell(self.ctx.discharged_unitcell_relaxed, dist)
                if len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell_relaxed.sites) == len(self.ctx.discharged_supercell_relaxed.sites) / len(self.ctx.discharged_unitcell_relaxed.sites): 
                    self.report(f'Distance<{distance}> given to build supercells was not appropriate, building supercells with new distance<{dist}>')
                    break

        elif len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell_relaxed.sites) < len(self.ctx.discharged_supercell_relaxed.sites) / len(self.ctx.discharged_unitcell_relaxed.sites):
            for dist in np.arange(distance, distance_upperbound, distance_epsilon):
                self.ctx.charged_supercell_relaxed = func.make_supercell(self.ctx.charged_unitcell_relaxed, dist)
                self.ctx.discharged_supercell_relaxed = func.make_supercell(self.ctx.discharged_unitcell_relaxed, distance)
                if len(self.ctx.charged_supercell_relaxed.sites) / len(self.ctx.charged_unitcell_relaxed.sites) == len(self.ctx.discharged_supercell_relaxed.sites) / len(self.ctx.discharged_unitcell_relaxed.sites): 
                    self.report(f'Distance<{distance}> given to build supercells was not appropriate, building supercells with new distance<{dist}>')
                    break

        self.ctx.charged_supercell_relaxed.set_extra('relaxed', True)
        self.ctx.charged_supercell_relaxed.set_extra('supercell', True)

        self.ctx.discharged_supercell_relaxed.set_extra('relaxed', True)
        self.ctx.discharged_supercell_relaxed.set_extra('supercell', True)

        res = func.get_unique_cation_sites(self.ctx.discharged_supercell_relaxed, self.ctx.cation)
        all_cation_indices, unique_cation_indices = res['all_cation_indices'], res['unique_cation_indices']
        self.ctx.low_SOC_supercells_d = func.get_low_SOC(self.ctx.discharged_supercell_relaxed, unique_cation_indices) 
        self.ctx.high_SOC_supercells_d = func.get_high_SOC(self.ctx.discharged_supercell_relaxed, self.ctx.charged_supercell_relaxed, all_cation_indices, unique_cation_indices)
        self.ctx.total_cations_supercell = len(all_cation_indices)

        return

    def run_relax_low_SOC(self):
        """
        Runs a PwRelaxWorkChain to relax the supercells with one cation removed.
        """
        inputs = self.ctx.ocv_relax
        struct = self.ctx.low_SOC_supercells_d['low_SOC_structure_00']

        if self.ctx.ocv_parameters_d['SOC_relax_all_supercells']:
            raise NotImplementedError('Relaxing all low SOC supercells is not implemented yet')
        else:
            inputs['structure'] = struct

        if not self.ctx.ocv_parameters_d['SOC_vc_relax']:
            inputs.base.pw.parameters['CONTROL']['calculation'] = 'relax'
            inputs.base.pw.parameters.pop('CELL')

        # Readding the Li pseudo back in
        inputs['base']['pw']['pseudos']['Li'] = self.ctx.Li_pseudo
        inputs['base_final_scf']['pw']['pseudos']['Li'] = self.ctx.Li_pseudo

        inputs.base.pw.parameters['SYSTEM']['tot_charge'] = float(-struct.extras['missing_cations'])
        inputs.base_final_scf.pw.parameters['SYSTEM']['tot_charge'] = float(-struct.extras['missing_cations'])

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
        To do - run this relax chain in parallel with previous one.
        """
        inputs = self.ctx.ocv_relax
        struct = self.ctx.high_SOC_supercells_d['high_SOC_structure_00']

        if self.ctx.ocv_parameters_d['SOC_relax_all_supercells']:
            raise NotImplementedError('Relaxing all low SOC supercells is not implemented yet')
        else:
            inputs['structure'] = struct

        if not self.ctx.ocv_parameters_d['SOC_vc_relax']:
            inputs.base.pw.parameters['CONTROL']['calculation'] = 'relax'
        
        inputs.base.pw.parameters['SYSTEM']['tot_charge'] = float(-struct.extras['missing_cations'])
        inputs.base_final_scf.pw.parameters['SYSTEM']['tot_charge'] = float(-struct.extras['missing_cations'])

        # Set the `CALL` link label
        self.inputs.metadata.call_link_label = 'high_SOC_relax'
        inputs.metadata.label = 'high_SOC_relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}>')
        
        return ToContext(relax_workchains=append_(running))
        
    def results(self):
        """
        Return the OCVs at various states of charge coefficients generated in the last MD run.
        """
        try:
            low_SOC_d = self.ctx.relax_workchains[-2].outputs.output_parameters.get_dict()
            high_SOC_d = self.ctx.relax_workchains[-1].outputs.output_parameters.get_dict()
        except exceptions.NotExistent:
            self.report('the PwRelaxWorkChains did not generate output parameters')
            return self.exit_codes.ERROR_DFT_ENERGY_NOT_FOUND
        try:
            charged_d = self.ctx.relax_workchains[-3].outputs.output_parameters.get_dict()
            discharged_d = self.ctx.relax_workchains[-4].outputs.output_parameters.get_dict()
        except AttributeError:
            discharged_d, charged_d = {}, {}
            try:
                discharged_d['energy'] = self.ctx.ocv_parameters_d['discharged_energy']
                charged_d['energy'] = self.ctx.ocv_parameters_d['charged_energy']
                self.report('Using the DFT energies of relaxed charged and dicharged structures given as input')
            except KeyError:
                self.report('DFT energies of relaxed structures not provided')
                return self.exit_codes.ERROR_DFT_ENERGY_NOT_FOUND
        
        # need to change the way to load cation energy and z when making this workchain for any general cation
        cation_energy = self.ctx.ocv_parameters_d[f'DFT_energy_bulk_{self.ctx.ocv_parameters_d["cation"]}']
        z = 1
        if self.ctx.ocv_parameters_d['cation'] == 'Mg': z = 2
        # normalising wrt cations
        total_cations_unitcell = self.ctx.charged_unitcell.extras['missing_cations']
        total_cations_supercell = self.ctx.total_cations_supercell

        V_low_SOC = -((discharged_d['energy'] / total_cations_unitcell) - (low_SOC_d['energy'] / total_cations_supercell) - cation_energy ) / z
        V_high_SOC = -((high_SOC_d['energy'] / total_cations_supercell) - (charged_d['energy'] / total_cations_unitcell) - cation_energy ) / z
        V_average = -((discharged_d['energy'] / total_cations_unitcell) - (charged_d['energy'] / total_cations_unitcell) - cation_energy ) / z

        ocv = orm.Dict(dict={'OCV_avergae': V_average, 'OCV_low_OCV': V_low_SOC, 'OCV_high_SOC': V_high_SOC, 'OCV_units': 'V'})

        self.out('open_circuit_voltages', ocv)
