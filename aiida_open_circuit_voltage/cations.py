# -*- coding: utf-8 -*-
"""Shared cation metadata and inference helpers."""

SUPPORTED_CATIONS = {
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Mg": 2,
    "Ca": 2,
    "Al": 3,
}


def is_missing_cation(cation):
    """Return True when a cation value should be inferred."""
    return cation is None or (isinstance(cation, str) and not cation.strip())


def supported_cation_labels():
    """Return supported cation labels in a stable display order."""
    return tuple(SUPPORTED_CATIONS)


def validate_cation(cation):
    """Validate and normalize an explicit cation label."""
    if is_missing_cation(cation):
        raise ValueError(
            "No cation was provided. Set ocv_parameters['cation'] or provide a "
            "structure containing exactly one supported cation."
        )

    cation = str(cation).strip()
    if cation not in SUPPORTED_CATIONS:
        raise ValueError(
            f"Cation '{cation}' is not supported. Choose from "
            f"{', '.join(supported_cation_labels())}."
        )
    return cation


def get_cation_valence(cation):
    """Return the ionic valence used in the OCV denominator."""
    return SUPPORTED_CATIONS[validate_cation(cation)]


def infer_cation_from_symbols(symbols):
    """Infer the cation when exactly one supported cation appears in symbols."""
    symbol_set = set(symbols)
    candidates = [
        cation for cation in supported_cation_labels() if cation in symbol_set
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            "Could not infer the cation: the structure contains none of the "
            f"supported cations ({', '.join(supported_cation_labels())}). "
            "Set ocv_parameters['cation'] explicitly."
        )

    raise ValueError(
        "Could not infer the cation because the structure contains multiple "
        f"supported candidates ({', '.join(candidates)}). Set "
        "ocv_parameters['cation'] explicitly."
    )


def infer_cation_from_aiida_structure(structure):
    """Infer the cation from an AiiDA StructureData-like object."""
    kinds = {kind.name: kind for kind in structure.kinds}
    symbols = []
    for site in structure.sites:
        kind = kinds[site.kind_name]
        kind_symbols = kind.symbols
        if isinstance(kind_symbols, str):
            kind_symbols = (kind_symbols,)
        if len(kind_symbols) == 1:
            symbols.append(kind_symbols[0])
    return infer_cation_from_symbols(symbols)


def infer_cation_from_ase_atoms(atoms):
    """Infer the cation from an ASE Atoms-like object."""
    return infer_cation_from_symbols(atom.symbol for atom in atoms)
