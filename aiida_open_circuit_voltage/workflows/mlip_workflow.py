# -*- coding: utf-8 -*-
"""
Standalone OCV workflow using ASE-compatible calculators.

"""
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


class MLIPOCVWorkflow:
    """
    Calculates open circuit voltages (OCV) at various states of charge using
    any ASE-compatible calculator (e.g. PET-MAD, MACE, etc).

    Mirrors the step-by-step logic of OCVWorkChain in the AiiDA plugin:

    1. Single-point energy of the bulk cation reference structure.
    2. Variable-cell relaxation of the discharged (fully cationated) unitcell.
    3. Variable-cell relaxation of the charged (fully decationated) unitcell.
    4. Mechanical-stability check and supercell construction.
    5. Fixed-cell (or vc-) relaxation of the low-SOC supercell
       (discharged supercell with 1 cation removed).
    6. Fixed-cell (or vc-) relaxation of the constrained charged unitcell
       (discharged lattice scaled to charged volume, no cations).
    7. Fixed-cell (or vc-) relaxation of the high-SOC supercell
       (1 cation remaining, lattice scaled to charged volume).
    8. Compute and return OCV values.

    Parameters
    structure : ase.Atoms
        Fully discharged (fully cationated) unitcell.
    calculator : ase.calculators.abc.AbcCalculator
        Loaded ASE calculator instance.
        The same instance is reused across all calculations.
    cation : str
        Element symbol of the intercalating ion. Supported: 'Li', 'Mg'.
    bulk_cation_structure : ase.Atoms
        Bulk reference structure for the cation (e.g. Li, Mg).
        Its energy per atom is used as the cation chemical potential.
        Should be equilibrium geometry, single-point energy is evaluated 
        (no relaxation is performed).
    distance : float
        Minimum supercell dimension in Angstrom passed to supercellor.
        Default: 8.0.
    fmax : float
        Force convergence threshold in eV/Ang for all ASE optimisers.
        Default: 0.05.
    steps : int
        Maximum number of optimisation steps per relaxation.
        Default: 500.
    optimiser : ASE optimiser class, optional
        Defaults to ``ase.optimize.BFGS``.
    cell_filter : ASE filter class, optional
        Wraps atoms for variable-cell relaxation.
        Defaults to ``ase.filters.ExpCellFilter``.
    SOC_vc_relax : bool
        If True, SOC supercells are variable-cell relaxed.
        If False (default), only atomic positions are relaxed.
    SOC_relax_all_supercells : bool
        If True, all symmetry-inequivalent SOC supercells are relaxed and
        the most stable is selected.
        If False (default), only the first unique supercell is used.
    volume_change_stability : bool
        Abort if the fractional volume change on charging exceeds the
        threshold. Default: True.
    volume_change_stability_threshold : float
        Maximum allowed |delta V/V|. Default: 0.1.
    """

    SUPPORTED_CATIONS = {"Li": 1, "Mg": 2}

    def __init__(
        self,
        structure,
        calculator,
        cation="Li",
        bulk_cation_structure=None,
        distance=8.0,
        fmax=0.05,
        steps=500,
        optimiser=None,
        cell_filter=None,
        SOC_vc_relax=False,
        SOC_relax_all_supercells=False,
        volume_change_stability=True,
        volume_change_stability_threshold=0.1,
    ):
        if cation not in self.SUPPORTED_CATIONS:
            raise ValueError(
                f"Cation '{cation}' not supported. Choose from {list(self.SUPPORTED_CATIONS)}."
            )
        if bulk_cation_structure is None:
            raise ValueError(
                "bulk_cation_structure is required. Provide an ASE Atoms object "
                "of the bulk cation reference (e.g. Li, Mg)."
            )

        if optimiser is None:
            from ase.optimize import BFGS
            optimiser = BFGS

        if cell_filter is None:
            from ase.filters import ExpCellFilter
            cell_filter = ExpCellFilter

        self.structure = structure.copy()
        self.calculator = calculator
        self.cation = cation
        self.z = self.SUPPORTED_CATIONS[cation]
        self.bulk_cation_structure = bulk_cation_structure.copy()
        self.distance = distance
        self.fmax = fmax
        self.steps = steps
        self.optimiser = optimiser
        self.cell_filter = cell_filter
        self.SOC_vc_relax = SOC_vc_relax
        self.SOC_relax_all_supercells = SOC_relax_all_supercells
        self.volume_change_stability = volume_change_stability
        self.volume_change_stability_threshold = volume_change_stability_threshold

    # Public interface

    def run(self):
        """
        Execute the full OCV workflow.

        Returns
        -------
        dict
            Keys:
            - ``OCV_average`` (float, V)
            - ``OCV_low_SOC`` (float or None, V)
            - ``OCV_high_SOC`` (float or None, V)
            - ``OCV_units`` (str, 'V')
            - ``discharged_relaxed`` (ase.Atoms)
            - ``charged_relaxed`` (ase.Atoms)
            - ``low_SOC_relaxed`` (ase.Atoms or None)
            - ``high_SOC_relaxed`` (ase.Atoms or None)
            - ``constrained_charged_relaxed`` (ase.Atoms or None)
        """

        # Step 1: Bulk cation reference energy (single-point)
        self._log("Step 1: Computing bulk cation reference energy (single-point).")
        cation_energy_per_atom = self._get_cation_energy()
        self._log(f" - Cation energy per atom: {cation_energy_per_atom:.6f} eV")

        # Step 2: Relax discharged unitcell (vc-relax)
        self._log("Step 2: Relaxing discharged unitcell (vc-relax).")
        discharged_relaxed, E_discharged = self._relax(self.structure, vc_relax=True)
        self._log(f" - E_discharged = {E_discharged:.6f} eV")

        # Step 3: Relax charged unitcell (vc-relax)
        self._log("Step 3: Relaxing charged unitcell (vc-relax).")
        charged_input = self._remove_cations(self.structure)
        charged_relaxed, E_charged = self._relax(charged_input, vc_relax=True)
        self._log(f" - E_charged    = {E_charged:.6f} eV")

        # Step 4: Mechanical stability check and supercell construction
        self._log("Step 4: Checking mechanical stability and building supercells.")
        V_discharged = discharged_relaxed.get_volume()
        V_charged = charged_relaxed.get_volume()
        volume_change = (V_charged - V_discharged) / V_discharged
        self._log(f" - Volume change on charging: {volume_change:+.3%}")

        if self.volume_change_stability:
            if abs(volume_change) > self.volume_change_stability_threshold:
                raise RuntimeError(
                    f"Structure is mechanically unstable: volume change {volume_change:.3%} "
                    f"exceeds threshold {self.volume_change_stability_threshold:.0%}."
                )

        discharged_supercell = self._make_supercell(discharged_relaxed)
        all_indices, unique_indices = self._get_unique_cation_sites(discharged_supercell)
        N_cations_unitcell = sum(1 for a in self.structure if a.symbol == self.cation)
        N_cations_supercell = len(all_indices)
        sc_scale = N_cations_supercell / N_cations_unitcell
        self._log(
            f" - Supercell: {len(discharged_supercell)} atoms, "
            f"{N_cations_supercell} {self.cation} sites, "
            f"{len(unique_indices)} unique."
        )

        # Step 5: Low SOC
        self._log("Step 5: Relaxing low-SOC supercell(s) (1 cation removed).")
        low_SOC_structures = self._get_low_SOC_structures(discharged_supercell, unique_indices)

        if self.SOC_relax_all_supercells:
            results_low = [
                self._relax(s, vc_relax=self.SOC_vc_relax) for s in low_SOC_structures
            ]
            low_SOC_relaxed, E_low = min(results_low, key=lambda x: x[1])
        else:
            low_SOC_relaxed, E_low = self._relax(
                low_SOC_structures[0], vc_relax=self.SOC_vc_relax
            )

        self._log(f" - E_low_SOC    = {E_low:.6f} eV")
        # V = [E(x-1) - (N_sc/N_uc)*E(discharged) + E_cation] / z
        OCV_low_SOC = (E_low - sc_scale * E_discharged + cation_energy_per_atom) / self.z

        # Step 6: Constrained charged unitcell
        constrained_charged_relaxed = None
        E_constrained = None

        self._log("Step 6: Relaxing constrained charged unitcell.")
        constrained_input = self._get_constrained_charged(discharged_relaxed, V_charged)
        constrained_charged_relaxed, E_constrained = self._relax(
            constrained_input, vc_relax=self.SOC_vc_relax
        )
        self._log(f" - E_constrained = {E_constrained:.6f} eV")

        # Step 7: High SOC
        self._log("Step 7: Relaxing high-SOC supercell(s) (1 cation remaining).")
        scaling_factor = V_charged / V_discharged
        new_volume = scaling_factor * discharged_supercell.get_volume()
        high_SOC_structures = self._get_high_SOC_structures(
            discharged_supercell, new_volume, all_indices, unique_indices
        )

        if self.SOC_relax_all_supercells:
            results_high = [
                self._relax(s, vc_relax=self.SOC_vc_relax) for s in high_SOC_structures
            ]
            high_SOC_relaxed, E_high = min(results_high, key=lambda x: x[1])
        else:
            high_SOC_relaxed, E_high = self._relax(
                high_SOC_structures[0], vc_relax=self.SOC_vc_relax
            )

        self._log(f" - E_high_SOC   = {E_high:.6f} eV")
        # V = [(N_sc/N_uc)*E(constrained) - E(x+1) + E_cation] / z
        OCV_high_SOC = (sc_scale * E_constrained - E_high + cation_energy_per_atom) / self.z

        # Step 8: Average OCV
        self._log("Step 8: Computing OCV values.")
        # V = [E(charged) - E(discharged) + N*E_cation] / (z*N)
        OCV_average = (
            E_charged - E_discharged + N_cations_unitcell * cation_energy_per_atom
        ) / (self.z * N_cations_unitcell)

        self._log("\nResults:")
        self._log(f"OCV average: {OCV_average:+.4f} V")
        self._log(f"OCV low SOC: {OCV_low_SOC:+.4f} V")
        self._log(f"OCV high SOC: {OCV_high_SOC:+.4f} V")

        return {
            "OCV_average": OCV_average,
            "OCV_low_SOC": OCV_low_SOC,
            "OCV_high_SOC": OCV_high_SOC,
            "OCV_units": "V",
            "discharged_relaxed": discharged_relaxed,
            "charged_relaxed": charged_relaxed,
            "low_SOC_relaxed": low_SOC_relaxed,
            "high_SOC_relaxed": high_SOC_relaxed,
            "constrained_charged_relaxed": constrained_charged_relaxed,
        }

    # Private helpers

    def _log(self, msg):
        print(msg)

    def _attach_calculator(self, atoms):
        """Return a copy of atoms with the calculator attached."""
        atoms = atoms.copy()
        atoms.calc = self.calculator
        return atoms

    def _relax(self, atoms, vc_relax=True):
        """
        Relax an ASE Atoms object and return (relaxed_atoms, energy).
        Uses cell_filter to do vc-relax 
        """
        atoms = self._attach_calculator(atoms)

        if vc_relax:
            filtered = self.cell_filter(atoms)
            opt = self.optimiser(filtered, logfile=None)
        else:
            opt = self.optimiser(atoms, logfile=None)

        opt.run(fmax=self.fmax, steps=self.steps)
        energy = atoms.get_potential_energy()
        return atoms, energy

    def _get_cation_energy(self):
        """
        Return the energy per atom of the bulk cation reference structure
        (single-point, no relaxation).
        """
        atoms = self._attach_calculator(self.bulk_cation_structure)
        energy = atoms.get_potential_energy()
        return energy / len(atoms)

    def _remove_cations(self, atoms):
        """Return a copy of atoms with all cation species removed."""
        indices = [a.index for a in atoms if a.symbol == self.cation]
        result = atoms.copy()
        del result[indices]
        return result

    def _make_supercell(self, atoms):
        """Build a supercell using supercellor and return as ASE Atoms."""
        from supercellor import supercell as sc

        pym_struct = AseAtomsAdaptor.get_structure(atoms)
        pym_sc, _ = sc.make_supercell(
            pym_struct, self.distance, verbosity=0, do_niggli_first=False
        )
        return AseAtomsAdaptor.get_atoms(pym_sc)

    def _get_unique_cation_sites(self, supercell):
        """
        Return (all_cation_indices, unique_cation_indices) in the supercell.

        Uses pymatgen SpacegroupAnalyzer (symprec=1e-5) to identify
        symmetry inequivalent cation sites, matching the same algorithm
        used in the main OCV workflow.
        """
        pym_struct = AseAtomsAdaptor.get_structure(supercell)
        cation_sites = [s for s in pym_struct.sites if s.species_string == self.cation]
        symmops = SpacegroupAnalyzer(
            pym_struct, symprec=1e-5
        ).get_space_group_operations()

        unique_sites = [cation_sites[0]]
        for site in cation_sites[1:]:
            if not any(
                symmops.are_symmetrically_equivalent([site], [u], symm_prec=1e-5)
                for u in unique_sites
            ):
                unique_sites.append(site)

        all_indices = [a.index for a in supercell if a.symbol == self.cation]
        unique_indices = [
            a.index
            for a in supercell
            for u in unique_sites
            if a.symbol == self.cation
            and (np.around(a.position, 5) == np.around(u.coords, 5)).all()
        ]
        return all_indices, unique_indices

    def _get_low_SOC_structures(self, supercell, unique_indices):
        """
        Return a list of ASE Atoms with 1 cation removed at each unique site.
        """
        structures = []
        for idx in unique_indices:
            s = supercell.copy()
            del s[idx]
            structures.append(s)
        return structures

    def _get_constrained_charged(self, discharged_relaxed, new_volume):
        """
        Remove all cations from the relaxed discharged unitcell and scale
        its lattice to match the charged cell volume.
        """
        pym = AseAtomsAdaptor.get_structure(discharged_relaxed)
        pym.remove_species([self.cation])
        pym.scale_lattice(new_volume)
        return AseAtomsAdaptor.get_atoms(pym)

    def _get_high_SOC_structures(self, supercell, new_volume, all_indices, unique_indices):
        """
        Return a list of ASE Atoms with only 1 cation remaining at each
        unique site, with the lattice scaled to the charged cell volume.

        The pymatgen site indices match the ASE atom indices because
        AseAtomsAdaptor preserves atom ordering.
        """
        pym = AseAtomsAdaptor.get_structure(supercell)
        pym.scale_lattice(new_volume)

        structures = []
        for keep_idx in unique_indices:
            to_remove = [i for i in all_indices if i != keep_idx]
            high = pym.copy()
            high.remove_sites(to_remove)
            structures.append(AseAtomsAdaptor.get_atoms(high))
        return structures
