# -*- coding: utf-8 -*-
"""Helpers and calcfunctions for DFT+U+V (extended Hubbard) support in ``OCVWorkChain``.

This module is only exercised when the workchain runs in *Hubbard mode*. The plain-GGA
path never imports from here at runtime (the workchain only touches these functions when a
``hubbard_sc`` namespace is present), so plain-DFT behaviour is completely unaffected.

The Hubbard parameter flow is:

* the user supplies a *spec* (``{'U': [[kind, manifold], ...], 'V': [[kind, man, nbr, nbr_man], ...]}``)
  or a pre-initialised ``HubbardStructureData`` from which the spec is extracted;
* :func:`build_initialized_hubbard_structure` seeds a ``HubbardStructureData`` with placeholder
  (1e-8 eV) values which the ``SelfConsistentHubbardWorkChain`` converges with ``hp.x``;
* converged values are transferred to the SOC supercells either *exactly*
  (:func:`get_hubbard_supercell`, low-SOC) or as *per-symbol averages*
  (:func:`get_hubbard_averaged`, charged-composition cells).
"""
from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

HubbardStructureData = DataFactory("quantumespresso.hubbard_structure")

# Extras propagated from a source structure onto a derived Hubbard structure so the existing
# provenance/extras conventions of the plugin keep working unchanged.
_EXTRA_KEYS = (
    "original_unitcell",
    "structure_type",
    "missing_cations",
    "relaxed",
    "supercell",
)


class HubbardSpecError(ValueError):
    """Raised when a Hubbard spec is missing or malformed (workchain exit 207)."""


class CationIsHubbardAtomError(ValueError):
    """Raised when the OCV cation is itself a Hubbard atom/neighbour (workchain exit 208)."""


def _as_dict(spec):
    """Return a plain ``dict`` from either an ``orm.Dict`` or a python ``dict``."""
    if isinstance(spec, orm.Dict):
        return spec.get_dict()
    if isinstance(spec, dict):
        return spec
    raise HubbardSpecError("The Hubbard spec must be a dict or an orm.Dict.")


def validate_hubbard_spec(spec, structure=None, cation=None):
    """Validate and normalize a Hubbard spec.

    :param spec: dict / ``orm.Dict`` of the form 
        ``{'U': [[kind, manifold], ...], 'V': [[kind, man, nbr, nbr_man], ...]}``.
        ``V`` is optional; ``U`` must contain at least one entry.
    :param structure: optional ``StructureData`` used to check that every named kind exists.
    :param cation: optional cation label; raises if the cation appears as a Hubbard atom/neighbour.
    :returns: a normalized JSON-serializable spec dict with ``'U'`` and ``'V'`` lists.
    :raises HubbardSpecError: if the spec is malformed or references missing kinds.
    :raises CationIsHubbardAtomError: if the cation is a Hubbard atom or neighbour.
    """
    spec = _as_dict(spec)

    raw_u = spec.get("U") or []
    raw_v = spec.get("V") or []

    if not raw_u:
        raise HubbardSpecError("The Hubbard spec must contain at least one onsite 'U' entry (e.g. {'U': [['Mn', '3d']]}).")

    normalized_u = []
    for entry in raw_u:
        if len(entry) != 2:
            raise HubbardSpecError(f"Each 'U' entry must be [kind, manifold]; got {entry!r}.")
        normalized_u.append([str(entry[0]), str(entry[1])])

    normalized_v = []
    for entry in raw_v:
        if len(entry) != 4:
            raise HubbardSpecError(f"Each 'V' entry must be [kind, manifold, neighbour, neighbour_manifold]; got {entry!r}.")
        normalized_v.append([str(entry[0]), str(entry[1]), str(entry[2]), str(entry[3])])

    if structure is not None:
        kind_names = {kind.name for kind in structure.kinds}
        referenced = {e[0] for e in normalized_u}
        referenced |= {e[0] for e in normalized_v} | {e[2] for e in normalized_v}
        missing = referenced - kind_names
        if missing:
            raise HubbardSpecError(f"The Hubbard spec references kinds {sorted(missing)} that are not in the structure (available kinds: {sorted(kind_names)}).")

    if cation is not None:
        cation = str(cation)
        hubbard_kinds = {e[0] for e in normalized_u}
        hubbard_kinds |= {e[0] for e in normalized_v} | {e[2] for e in normalized_v}
        if cation in hubbard_kinds:
            raise CationIsHubbardAtomError(f"The cation '{cation}' appears as a Hubbard atom/neighbour in the spec. The cation is removed when building charged structures, which would invalidate its Hubbard parameters; choose a non-cation manifold.")

    return {"U": normalized_u, "V": normalized_v}


def _index_symbol_map(structure):
    """Return a list mapping each site index of ``structure`` to its chemical symbol."""
    kind_symbol = {kind.name: kind.symbol for kind in structure.kinds}
    return [kind_symbol[site.kind_name] for site in structure.sites]


def extract_hubbard_spec(hubbard_structure):
    """Extract a normalized spec from an initialised/converged ``HubbardStructureData``.

    Onsite parameters (``atom_index == neighbour_index`` with equal manifolds) become ``'U'``
    entries; the rest become ``'V'`` entries. Entries are keyed by *kind name* and de-duplicated
    while preserving order.
    """
    sites = hubbard_structure.sites
    seen_u, seen_v = set(), set()
    u_entries, v_entries = [], []

    for param in hubbard_structure.hubbard.parameters:
        kind_i = sites[param.atom_index].kind_name
        kind_j = sites[param.neighbour_index].kind_name
        is_onsite = (param.atom_index == param.neighbour_index and param.atom_manifold == param.neighbour_manifold)
        if is_onsite:
            key = (kind_i, param.atom_manifold)
            if key not in seen_u:
                seen_u.add(key)
                u_entries.append([kind_i, param.atom_manifold])
        else:
            key = (kind_i, param.atom_manifold, kind_j, param.neighbour_manifold)
            if key not in seen_v:
                seen_v.add(key)
                v_entries.append([kind_i, param.atom_manifold, kind_j, param.neighbour_manifold])

    if not u_entries:
        raise HubbardSpecError("The provided HubbardStructureData has no onsite (U) parameters to extract.")
    return {"U": u_entries, "V": v_entries}


def build_initialized_hubbard_structure(structure, spec):
    """Return an (unstored) ``HubbardStructureData`` seeded with placeholder (1e-8) U/V values.

    The real values are computed self-consistently by the ``SelfConsistentHubbardWorkChain``.
    """
    spec = validate_hubbard_spec(spec, structure=structure)
    hubbard_structure = HubbardStructureData.from_structure(structure)
    for kind, manifold in spec["U"]:
        hubbard_structure.initialize_onsites_hubbard(kind, manifold, 1e-8, "U")
    for kind, manifold, neighbour, neighbour_manifold in spec["V"]:
        hubbard_structure.initialize_intersites_hubbard(kind, manifold, neighbour, neighbour_manifold, 1e-8, "V")
    return hubbard_structure


def average_hubbard_by_symbol(hubbard_structure):
    """Average converged U/V values by chemical symbol (robust to kind relabelling).

    :returns: a tuple ``(u_avg, v_avg)`` where
        ``u_avg`` maps ``(symbol, manifold) -> averaged U`` and
        ``v_avg`` maps ``(symbol_i, man_i, symbol_j, man_j) -> averaged V``.
    """
    symbols = _index_symbol_map(hubbard_structure)
    u_acc, v_acc = {}, {}

    for param in hubbard_structure.hubbard.parameters:
        symbol_i = symbols[param.atom_index]
        symbol_j = symbols[param.neighbour_index]
        is_onsite = (param.atom_index == param.neighbour_index and param.atom_manifold == param.neighbour_manifold)
        if is_onsite:
            u_acc.setdefault((symbol_i, param.atom_manifold), []).append(param.value)
        else:
            key = (symbol_i, param.atom_manifold, symbol_j, param.neighbour_manifold)
            v_acc.setdefault(key, []).append(param.value)

    u_avg = {key: sum(values) / len(values) for key, values in u_acc.items()}
    v_avg = {key: sum(values) / len(values) for key, values in v_acc.items()}
    return u_avg, v_avg


def _build_averaged_hubbard_structure(structure, hubbard_source, spec):
    """Re-initialize Hubbard parameters on ``structure`` using per-symbol averages.

    Nearest-neighbour V couples are found with pymatgen (``CrystalNN``) via
    :func:`aiida_quantumespresso.utils.hubbard.initialize_hubbard_parameters`, then each value
    is overwritten with the per-symbol average taken from ``hubbard_source``. 
    Returns an unstored ``HubbardStructureData``.
    """
    from aiida_quantumespresso.common.hubbard import Hubbard
    from aiida_quantumespresso.utils.hubbard import initialize_hubbard_parameters

    spec = validate_hubbard_spec(spec)
    u_avg, v_avg = average_hubbard_by_symbol(hubbard_source)

    # Map each onsite Hubbard symbol to its manifold, U average, a placeholder V, and the set of
    # neighbour symbols/manifolds from the spec. ``initialize_hubbard_parameters`` keys ``pairs``
    # by kind name; in pymatgen-derived charged structures kind name == chemical symbol, so we key
    # by symbol throughout.
    source_symbols = {kind.name: kind.symbol for kind in hubbard_source.kinds}

    pairs = {}
    for kind, manifold in spec["U"]:
        symbol = source_symbols.get(kind, kind)
        neighbours = {}
        for v_kind, v_man, v_nbr, v_nbr_man in spec["V"]:
            if v_kind == kind:
                neighbours[source_symbols.get(v_nbr, v_nbr)] = v_nbr_man
        u_value = u_avg.get((symbol, manifold), 1e-8)
        pairs[symbol] = (manifold, u_value, 1e-8, neighbours)

    target_symbols = {kind.name for kind in structure.kinds}
    pairs = {symbol: data for symbol, data in pairs.items() if symbol in target_symbols}

    hubbard_structure = initialize_hubbard_parameters(structure=structure, pairs=pairs)

    # Overwrite every initialised value with the matching per-symbol average. This handles
    # multiple neighbour kinds per onsite kind, which the single V slot in ``pairs`` cannot.
    symbols = _index_symbol_map(hubbard_structure)
    rebuilt = []
    for param in hubbard_structure.hubbard.parameters:
        symbol_i = symbols[param.atom_index]
        symbol_j = symbols[param.neighbour_index]
        is_onsite = (
            param.atom_index == param.neighbour_index
            and param.atom_manifold == param.neighbour_manifold
        )
        if is_onsite:
            value = u_avg.get((symbol_i, param.atom_manifold), param.value)
        else:
            value = v_avg.get(
                (symbol_i, param.atom_manifold, symbol_j, param.neighbour_manifold),
                param.value,
            )
        tup = list(param.to_tuple())
        tup[4] = value
        rebuilt.append(tuple(tup))

    hubbard = hubbard_structure.hubbard
    new_hubbard = Hubbard.from_list(rebuilt, hubbard.projectors, hubbard.formulation)
    return HubbardStructureData.from_structure(structure=structure, hubbard=new_hubbard)


def _copy_structure_extras(source, target, **overrides):
    """Copy the plugin's provenance extras from ``source`` onto ``target`` (plus overrides)."""
    source_extras = source.base.extras.all
    for key in _EXTRA_KEYS:
        if key in source_extras:
            target.set_extra(key, source_extras[key])
    for key, value in overrides.items():
        target.set_extra(key, value)



@calcfunction
def initialize_hubbard_structure(structure, hubbard_spec):
    """Return a ``HubbardStructureData`` seeded with placeholder U/V values for the SC loop."""
    hubbard_structure = build_initialized_hubbard_structure(structure, hubbard_spec)
    _copy_structure_extras(structure, hubbard_structure)
    return {"hubbard_structure": hubbard_structure}


@calcfunction
def get_hubbard_supercell(structure, hubbard_unitcell, hubbard_spec):
    """Transfer converged unitcell U/V onto a (possibly cation-depleted) supercell.

    Tries the *exact* geometric mapping (`HubbardUtils.get_hubbard_for_supercell`). If the
    mapping raises or fails to cover every Hubbard atom (e.g. because the supercell lattice was
    standardized/rotated), it falls back to the per-symbol averaged re-initialization. The chosen
    method is recorded in the ``hubbard_transfer_method`` extra (``'exact'`` / ``'averaged_fallback'``).
    """
    from aiida_quantumespresso.utils.hubbard import HubbardUtils

    spec = validate_hubbard_spec(hubbard_spec)
    method = "exact"
    hubbard_structure = None
    try:
        hubbard_structure = HubbardUtils(hubbard_unitcell).get_hubbard_for_supercell(structure)
        if not _supercell_fully_covered(hubbard_structure, spec):
            hubbard_structure = None
    except Exception:  # any mapping failure falls back to the averaged path
        hubbard_structure = None

    if hubbard_structure is None:
        method = "averaged_fallback"
        hubbard_structure = _build_averaged_hubbard_structure(structure, hubbard_unitcell, spec)

    _copy_structure_extras(structure, hubbard_structure, hubbard_transfer_method=method)
    return {"hubbard_structure": hubbard_structure}


@calcfunction
def get_hubbard_averaged(structure, hubbard_source, hubbard_spec):
    """Re-initialize ``structure`` with per-symbol-averaged U/V taken from ``hubbard_source``.

    Used for the charged-composition cells (constrained-charged unitcell and high-SOC
    supercells), which are scaled discharged supercells and therefore not commensurate with the
    relaxed charged unitcell, so exact mapping is impossible by construction.
    """
    spec = validate_hubbard_spec(hubbard_spec)
    hubbard_structure = _build_averaged_hubbard_structure(structure, hubbard_source, spec)
    _copy_structure_extras(structure, hubbard_structure, hubbard_transfer_method="averaged")
    return {"hubbard_structure": hubbard_structure}


def _supercell_fully_covered(supercell_hubbard, spec):
    """Return True if every supercell site of an onsite Hubbard symbol carries an onsite param."""
    onsite_symbols = {kind for kind, _ in spec["U"]}
    # ``spec`` uses unitcell kind names; supercell (pymatgen-derived) kinds equal symbols. Compare
    # on symbols so the check is robust to kind/symbol naming.
    symbol_kinds = {kind.name: kind.symbol for kind in supercell_hubbard.kinds}
    onsite_symbols = {symbol_kinds.get(name, name) for name in onsite_symbols}

    symbols = _index_symbol_map(supercell_hubbard)
    required = {index for index, symbol in enumerate(symbols) if symbol in onsite_symbols}
    covered = {
        param.atom_index for param in supercell_hubbard.hubbard.parameters
        if param.atom_index == param.neighbour_index
        and param.atom_manifold == param.neighbour_manifold
    }
    return bool(required) and required.issubset(covered)
