# -*- coding: utf-8 -*-
from aiida.engine import calcfunction
bohr_to_ang = 0.52917720859
timeau_to_sec = 2.418884254E-17

from aiida import orm
from aiida.engine import calcfunction

import numpy as np

SINGULAR_TRAJ_KEYS = ('symbols', 'atomic_species_name')

@calcfunction
def make_supercell(structure, distance):
    from supercellor import supercell as sc
    pym_sc_struct = sc.make_supercell(structure.get_pymatgen_structure(), distance, verbosity=0, do_niggli_first=False)[0]
    sc_struct = orm.StructureData()
    sc_struct.set_extra('original_unitcell', structure.uuid)
    sc_struct.set_extra('structure_type', 'discharged')
    sc_struct.set_pymatgen(pym_sc_struct)
    return sc_struct


@calcfunction
def get_low_SOC(structure, cations=['Li', 'Mg']):
    '''
    Returns a list of structures made after removing 1 symmeterically inquevalent cation 
    '''

    structure_ase = structure.get_ase()
    ## Make a list of all possible supercells with 1 cation removed
    low_SOC_structures = []
    for atom in structure_ase:
        if atom.symbol in cations:
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
        decationised_structure.set_extra('structure_type', 'low_SOC')
        decationised_structure.set_extra('missing_cations', 1)
        decationised_structure.set_ase(struct)
        decationised_structure.label = decationised_structure.get_formula(mode='count')
        unique_low_SOC_aiida_structures.append(decationised_structure)

    return unique_low_SOC_aiida_structures


@calcfunction
def get_charged(structure, cation_to_remove):
    """
    Take the input structure and build a completely charged structure i.e. 
    structure containing no cations
    """
    
    assert isinstance(structure, orm.StructureData), "input structure needs to be an instance of {}".format(orm.StructureData)

    struct_ase = structure.get_ase()
    cations_indices = [atom.index for atom in struct_ase if atom.symbol == cation_to_remove]
    del struct_ase[cations_indices]

    decationised_structure = orm.StructureData()
    decationised_structure.set_extra('original_unitcell', structure.uuid)
    decationised_structure.set_extra('structure_type', 'charged')
    decationised_structure.set_extra('missing_cations', len(cations_indices))
    decationised_structure.set_ase(struct_ase)
    decationised_structure.label = decationised_structure.get_formula(mode='count')

    return decationised_structure