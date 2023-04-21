# -*- coding: utf-8 -*-

from aiida import orm
from aiida.engine import calcfunction
import numpy as np

@calcfunction
def get_unique_cation_sites(structure, cation):
    '''
    Returns the indices of unique cationic positions in the structure 
    '''
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cation = cation.value
    structure_pym = structure.get_pymatgen_structure()
    structure_ase = structure.get_ase()
    cation_sites = [site for site in structure_pym.sites if site.species_string == cation]
    symmops = SpacegroupAnalyzer(structure_pym, symprec=1e-5).get_space_group_operations()

    # first site is always unique
    unique_sites = [cation_sites[0], ]
    for site_1 in cation_sites[1:]:
        unique = True
        for site_2 in unique_sites: 
            if symmops.are_symmetrically_equivalent([site_1,], [site_2,], symm_prec=1e-5): unique = False 
        if unique: unique_sites.append(site_1)
    
    all_cation_indices = [atom.index for atom in structure_ase if atom.symbol == cation]
    unique_cation_indices = [atom.index for atom in structure_ase for unique_site in unique_sites if (np.around(atom.position, 5) == np.around(unique_site.coords, 5)).all()]

    return {'all_cation_indices':orm.List(list=all_cation_indices), 'unique_cation_indices':orm.List(list=unique_cation_indices)}

def make_supercell(structure, distance):
    from supercellor import supercell as sc
    pym_sc_struct = sc.make_supercell(structure.get_pymatgen_structure(), distance, verbosity=0, do_niggli_first=False)[0]
    sc_struct = orm.StructureData()
    sc_struct.set_extra('original_unitcell', structure.uuid)
    sc_struct.set_extra('structure_type', 'discharged')
    sc_struct.set_pymatgen(pym_sc_struct)
    return sc_struct

## Using pymatgen symmetric site finder (much faster)
@calcfunction
def get_low_SOC(structure, unique_indices):
    '''
    Returns a list of structures made after removing 1 symmeterically inquevalent cation.
    We assume that removing one cation from discharged structure doesn't distort the cell.
    '''
    ## Make a list of unique (symmetrically inequivalent) supercells with 1 cation removed
    unique_indices = unique_indices.get_list()
    structure_ase = structure.get_ase()
    low_SOC_structures = []
    for idx in unique_indices:
        low_SOC = structure_ase.copy()
        del low_SOC[idx]
        low_SOC_structures.append(low_SOC)

    ## In case something went wrong with new structure generation
    assert len(low_SOC_structures)==len(unique_indices), f'{len(unique_indices)} unique sites identified by pymatgen, but {len(low_SOC_structures)} unique structures generated'

    ## Store the ase structures as aiida StructureData
    unique_low_SOC_aiida_structures = []
    for struct in low_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'low_SOC')
        decationised_structure.set_extra('missing_cations', 1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_low_SOC_aiida_structures.append(decationised_structure)

    return {f'low_SOC_structure_{idx:02d}': structure for idx, structure in enumerate(unique_low_SOC_aiida_structures)}

## Using ase symmetry comparison
@calcfunction
def get_low_SOC_slow(structure, cation):
    '''
    Returns a list of structures made after removing 1 symmeterically inquevalent cation.
    We assume that removing one cation from discharged structure doesn't distort the cell.
    '''
    from ase.utils.structure_comparator import SymmetryEquivalenceCheck

    cation = cation.value
    structure_ase = structure.get_ase()
    ## Make a list of all possible supercells with 1 cation removed
    low_SOC_structures = []
    for atom in structure_ase:
        if atom.symbol == cation:
            low_SOC = structure_ase.copy()
            del low_SOC[atom.index]
            low_SOC_structures.append(low_SOC)

    ## Make a list of unique (symmetrically inequivalent) supercells with 1 cation removed
    sym_eq_check = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05, stol=0.1, vol_tol=0.1, scale_volume=False, to_primitive=False)
    unique_low_SOC_structures = [low_SOC_structures[0], ]
    for struct_1 in low_SOC_structures[1:]:
        unique = True
        for struct_2 in unique_low_SOC_structures: 
            if (sym_eq_check.compare(struct_1, struct_2)): unique = False 
        if unique: unique_low_SOC_structures.append(struct_1)

    ## Store the ase structures as aiida StructureData
    unique_low_SOC_aiida_structures = []
    for struct in unique_low_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'high_SOC')
        decationised_structure.set_extra('missing_cations', 1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_low_SOC_aiida_structures.append(decationised_structure)

    return {f'low_SOC_structure_{idx:02d}': structure for idx, structure in enumerate(unique_low_SOC_aiida_structures)}

@calcfunction
def get_high_SOC(structure, scaling_factor, all_cation_indices, unique_indices):
    '''
    Returns a list of structures made after removing all but 1 symmeterically inquevalent cation
    i.e. a structure that contains only 1 cation.
    We assume that adding one cation to a completely charged structure doesn't distort the cell.
    Since it is easier to remove atoms than add one at specific sites, we scale the discharged 
    supercell wrt the lattice vectors of charged supercell and then remove all but one cation.
    '''
    all_cation_indices = all_cation_indices.get_list()
    unique_indices = unique_indices.get_list()
    structure_ase = structure.get_ase()

    structure_ase.set_cell(structure_ase.get_cell() * scaling_factor, scale_atoms=True)

    ## Make a list of all possible supercells with only 1 cation remaining
    high_SOC_structures = []
    for idx in unique_indices:
        tmp_indices = all_cation_indices.copy()
        high_SOC = structure_ase.copy()
        ## keeping all but one inequivalent cation
        tmp_indices.remove(idx)
        del high_SOC[tmp_indices]
        high_SOC_structures.append(high_SOC)
    ## In case somethign went wrong with new structure generation
    assert len(high_SOC_structures)==len(unique_indices), f'{len(unique_indices)} unique sites identified by pymatgen, but {len(high_SOC_structures)} unique structures generated'

    ## Store the ase structures as aiida StructureData
    unique_high_SOC_aiida_structures = []
    for struct in high_SOC_structures:
        decationised_structure = orm.StructureData()
        decationised_structure.set_extra('original_unitcell', structure.extras['original_unitcell'])
        decationised_structure.set_extra('structure_type', 'high_SOC')
        decationised_structure.set_extra('missing_cations', len(all_cation_indices)-1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_high_SOC_aiida_structures.append(decationised_structure)

    return {f'high_SOC_structure_{idx:02d}': structure for idx, structure in enumerate(unique_high_SOC_aiida_structures)}

@calcfunction
def get_charged(structure, cation_to_remove):
    """
    Take the input structure and build a completely charged structure i.e. 
    structure containing no cations
    """
    cation_to_remove = cation_to_remove.value
    struct_ase = structure.get_ase()
    cations_indices = [atom.index for atom in struct_ase if atom.symbol == cation_to_remove]
    del struct_ase[cations_indices]

    decationised_structure = orm.StructureData()
    decationised_structure.set_extra('original_unitcell', structure.uuid)
    decationised_structure.set_extra('structure_type', 'charged')
    decationised_structure.set_extra('missing_cations', len(cations_indices))
    decationised_structure.set_ase(struct_ase)
    decationised_structure.label = decationised_structure.get_formula(mode='count')

    return {'decationised_structure': decationised_structure}

def get_optimade(structure):
    
    ## to do - make it a class function when adding to aiida-core
    from aiida.orm.nodes.data.structure import Kind, Site

    if isinstance(structure, orm.StructureData):
        structure_ase = structure.get_ase()
    else:
        raise TypeError('structure type not valid')

    tmp_dicts = structure.attributes['kinds']
    for tmp_dict in tmp_dicts:
        tmp_dict.pop('weights')
        symbol = tmp_dict.pop('symbols')[0]
        tmp_dict['chemical_symbols'] = symbol

    optimade = orm.Dict(dict={'immutable_id': structure.uuid,
    'elements': structure.get_kind_names(),
    'chemical_formula_descriptive': structure.get_formula(),
    'dimension_types': np.multiply(structure.pbc, 1),
    'lattice_vectors': structure_ase.get_cell().tolist(),
    'cartesian_site_positions':structure_ase.get_positions().tolist(),
    'species': tmp_dicts,
    'species_at_sites': structure.get_site_kindnames(),
    'assemblies': None,
    'structure_features':[]})

    return optimade

def get_structuredata_from_optimade(structure, load_from_uuid=orm.Bool(False)):
    
    from aiida.orm.nodes.data.structure import Kind, Site

    if isinstance(structure, orm.Dict):
        structure_d = structure.get_dict()
    elif isinstance(structure, dict):
        structure_d = structure
    else:
        raise TypeError('structure type not valid')

    structure_aiida = orm.StructureData()

    uuid = structure_d['immutable_id']
    if uuid and load_from_uuid:
        try:
            structure_aiida = orm.load_node(uuid)
            structure_aiida.set_extra('queried_from_optimade', True)
            return structure_aiida
        except:
            raise AttributeError('The uuid does not exist in the database')
    elif load_from_uuid:
        structure_aiida.set_extra('generated_from_optimade', True)

    structure_aiida.set_cell(structure_d['lattice_vectors'])
    
    ## to do - add option to add kinds based on magnetisation treatment
    for kind in structure_d['species']:
        structure_aiida.append_kind(Kind(symbols=kind['chemical_symbols'][0], mass=kind['mass'][0], name=kind['name']))

    for specie, position in zip(structure_d['species_at_sites'], structure_d['cartesian_site_positions']):
        structure_aiida.append_site(Site(kind_name=specie, position=position))

    structure_aiida.store()

    return structure_aiida

@calcfunction
def get_OCVs(ocv_parameters, discharged_ouput_parameter, charged_ouput_parameter, low_SOC_ouput_parameter=None, high_SOC_ouput_parameter=None, bulk_cation_scf_output=None):
    """
    Take the output parameters containing DFT energies and calculated the OCV. 
    structure containing no cations
    :param discharged_ouput_parameter: the ``Dictionary`` instance output of the ``PwRelaxWorkChain`` run on ``discharged`` structure.
    :param charged_ouput_parameter: the ``Dictionary`` instance output of the ``PwRelaxWorkChain`` run on ``charged`` structure.
    :param low_SOC_ouput_parameter: the ``Dictionary`` instance output of the ``PwRelaxWorkChain`` run on ``low state of charge`` structure.
    :param high_SOC_ouput_parameter: the ``Dictionary`` instance output of the ``PwRelaxWorkChain`` run on ``high state of charge`` structure.
    :param ocv_parameters: the ``Dictionary`` instance used within the OCVWorkChain.
    :param bulk_cation_scf_output: the optional ``Dictionary`` instance output of the ``PwBaseWorkChain`` run on ``bulk cation`` structure, if this is not provided the energy values are read
    from the ocv_parameters.
    """
    from aiida.plugins import WorkflowFactory

    ocv_parameters_d =  ocv_parameters.get_dict()
    discharged_d = discharged_ouput_parameter.get_dict()
    charged_d = charged_ouput_parameter.get_dict()

    # Loading the charged structure
    charged_unitcell = charged_ouput_parameter.get_incoming(WorkflowFactory('quantumespresso.pw.relax')).all_nodes()[-1].inputs['structure']
    total_cations_unitcell = charged_unitcell.extras['missing_cations']

    if low_SOC_ouput_parameter:
        low_SOC_d = low_SOC_ouput_parameter.get_dict()
        low_SOC_supercell = low_SOC_ouput_parameter.get_incoming(WorkflowFactory('quantumespresso.pw.relax')).all_nodes()[-1].inputs['structure']
        low_SOC_ss_ase = low_SOC_supercell.get_ase()
        cations_indices_low = [atom.index for atom in low_SOC_ss_ase if atom.symbol == ocv_parameters_d['cation']]
        total_cations_supercell = len(cations_indices_low) + 1

    if high_SOC_ouput_parameter:
        high_SOC_d = high_SOC_ouput_parameter.get_dict()
        # Loading the high SOC structure
        high_SOC_supercell = high_SOC_ouput_parameter.get_incoming(WorkflowFactory('quantumespresso.pw.relax')).all_nodes()[-1].inputs['structure']
        total_cations_supercell = high_SOC_supercell.extras['missing_cations'] + 1

    if bulk_cation_scf_output:
        bulk_cation_d = bulk_cation_scf_output.get_dict()
        # Loading the bulk cation structure
        bulk_cation_structure = bulk_cation_scf_output.get_incoming(WorkflowFactory('quantumespresso.pw.base')).all_nodes()[-1].inputs['pw']['structure']
        # I add back the only cation present in the high SOC structure, to get total number of cations in the supercell
        cation_energy = bulk_cation_d['energy'] / len(bulk_cation_structure.sites)
    else:
        cation_energy = ocv_parameters_d[f'DFT_energy_bulk_{ocv_parameters_d["cation"]}']

    # need to change the way to load cation energy and z when making this workchain for any general cation
    if ocv_parameters_d['cation'] == 'Li': 
        z = 1
    elif ocv_parameters_d['cation'] == 'Mg': 
        z = 2
    else: 
        raise NotImplemented('Only Li and Mg ion materials supported now.')

    # I use the following standard equation for calculating voltage between 2 states with x1 and x2 concentration of Li atoms
    # voltage = -[E(Lix2) - E(Lix1) - [x2-x1]E(Li)] / [x2-x1]z
    if ocv_parameters_d['do_low_SOC_OCV']:
        # x2-x1 = 1 in this case
        V_low_SOC = (low_SOC_d['energy'] - (total_cations_supercell / total_cations_unitcell) * discharged_d['energy'] + 1 * cation_energy) / (z * 1) 
    else: 
        V_low_SOC = 'not_calculated'

    if ocv_parameters_d['do_high_SOC_OCV']:
        # x2-x1 = 1 in this case
        V_high_SOC = ((total_cations_supercell / total_cations_unitcell) * charged_d['energy'] - high_SOC_d['energy'] + 1 * cation_energy) / (z * 1)
    else: 
        V_high_SOC = 'not_calculated'
    # x2-x1 = all Li atoms in the discharged unitcell in this case
    V_average = ((charged_d['energy'] - discharged_d['energy'] + total_cations_unitcell * cation_energy) ) / (z * total_cations_unitcell)

    ocv = orm.Dict(dict={'OCV_avergae': V_average, 'OCV_low_OCV': V_low_SOC, 'OCV_high_SOC': V_high_SOC, 'OCV_units': 'V'})
    
    return ocv

@calcfunction
def get_json_outputs(ocv, discharged_structure, charged_structure, low_SOC_structure=None, high_SOC_structure=None, meta=None, optional_outputs=None):
    """
    Take the output parameters containing DFT energies and calculated the OCV. 
    structure containing no cations
    :param ocv: the ``Dictionary`` instance output of the ``OCVWorkChain`` calculated by the ``get_OCVs()`` calcfunction.
    :param discharged_structure: the ``StructureData`` instance output of the ``PwRelaxWorkChain`` run on ``discharged`` structure, i.e. the relaxed discharged structure.
    :param charged_structure: the ``StructureData`` instance output of the ``PwRelaxWorkChain`` run on ``charged`` structure, i.e. the relaxed charged structure.
    :param low_SOC_ouput_parameter: the ``StructureData`` instance output of the ``PwRelaxWorkChain`` run on ``low state of charge`` structure, i.e. the relaxed low SOC structure.
    :param high_SOC_ouput_parameter: the ``StructureData`` instance output of the ``PwRelaxWorkChain`` run on ``high state of charge`` structure, i.e. the relaxed high SOC structure.
    :param meta: the ``Dictionary`` instance used to output any ``meta`` outputs.
    :param optional_outputs: the ``Dictionary`` instance used to output any other remaining outputs like forces, stresses etc.
    """

    from aiida_open_circuit_voltage.calculations.functions import functions as func

    discharged_structure_o = func.get_optimade(discharged_structure).get_dict()
    charged_structure_o = func.get_optimade(charged_structure).get_dict()

    if low_SOC_structure:
        low_SOC_structure_o = {"low_SOC_structure_01": get_optimade(low_SOC_structure).get_dict()}
    else:
        low_SOC_structure_o = {"low_SOC_structure_01": None}

    if high_SOC_structure:
        high_SOC_structure_o = {"high_SOC_structure_01": get_optimade(high_SOC_structure).get_dict()}
    else:
        high_SOC_structure_o = {"high_SOC_structure_01": None}
    
    if not meta: meta = {}
    if not optional_outputs: optional_outputs = {}

    outputs = {"OCV_values_eV": ocv.get_dict(),
    "fully_discharged_structure": discharged_structure_o,
    "fully_charged_structure": charged_structure_o,
    "fully_charged_structure_with_discharged_cell": None,
    "high_SOC_structures": high_SOC_structure_o,
    "low_SOC_structures": low_SOC_structure_o,
    "optional_outputs": optional_outputs}    

    json_out = orm.Dict(dict={"task": "ocv", "outputs": outputs, "meta": meta})

    return json_out
    