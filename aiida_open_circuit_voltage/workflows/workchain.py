# -*- coding: utf-8 -*-
"""
Mother Workchain that calls PwRelaxWorkChain to relax experimental structures and
calculate DFT energies used to compute open circuit voltages (OCV) at low and high state of 
charge (SOC) and average OCV for any arbitrary cathode material
"""
import json
import numpy as np
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.engine import ToContext, append_, WorkChain, if_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.plugins import WorkflowFactory, DataFactory
from aiida_quantumespresso.common.types import SpinType
from aiida_open_circuit_voltage.cations import (
    infer_cation_from_aiida_structure,
    is_missing_cation,
    validate_cation,
)
from aiida_open_circuit_voltage.calculations.functions import functions as func
from aiida_open_circuit_voltage.calculations.functions import hubbard_functions as hub_func

PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")
PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
SelfConsistentHubbardWorkChain = WorkflowFactory("quantumespresso.hp.hubbard")
HubbardStructureData = DataFactory("quantumespresso.hubbard_structure")


class OCVWorkChain(ProtocolMixin, WorkChain):
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
    based on user input either do fixed cell relax of one such supercell or do fixed cell relax of all unique
    supercells and take the lowest formation energy

    High SOC
    scale the discharged supercell with the lattice vectors of charged supercell and remove all but 1 cation
    make list of sypercells after leaving 1 (symmetery inequivalent) cation
    based on user input either do fixed cell relax of one such supercell or do fixed cell relax of all unique
    supercells and take the lowest formation energy
    """

    @classmethod
    def define(cls, spec):
        """
        Define the process specification.
        """
        super().define(spec)
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="ocv_relax",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `PwRelaxWorkChain` for running the four relax calculations are called in the `ocv_relax` namespace."
            },
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="scf",
            exclude=("clean_workdir", "pw.structure"),
            namespace_options={
                "help": "Inputs for the `PwBaseWorkChain` for the SCF calculation of bulk cation and for if relaxed charged-discharged unitcells are provided.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            SelfConsistentHubbardWorkChain,
            namespace="hubbard_sc",
            exclude=("clean_workdir", "hubbard_structure"),
            namespace_options={
                "help": "Inputs for the `SelfConsistentHubbardWorkChain` run on the discharged and charged unitcells. Providing this namespace switches the workchain into DFT+U+V (Hubbard) mode.",
                "required": False,
                "populate_defaults": False,
            },
        )
        # The SelfConsistentHubbardWorkChain has a top-level inputs validator that assumes the scf
        # namespace is populated; null it here so an absent (plain-DFT) `hubbard_sc` namespace does
        # not raise. We re-validate the Hubbard inputs ourselves in `setup`.
        spec.inputs["hubbard_sc"].validator = None
        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="The input unitcell structure. May be a `HubbardStructureData`, in which case the Hubbard spec is inferred from it.",
        )
        spec.input(
            "bulk_cation_structure",
            valid_type=orm.StructureData,
            required=False,
            help="The input bulk cation structure.",
        )
        spec.input(
            "ocv_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The dictionary containing the parameters used to calculate OCVs.",
        )
        spec.input(
            "discharged_unitcell_relaxed",
            valid_type=orm.StructureData,
            required=False,
            help="The relaxed unitcell needed to restart this workchain.",
        )
        spec.input(
            "charged_unitcell_relaxed",
            valid_type=orm.StructureData,
            required=False,
            help="The relaxed unitcell needed to restart this workchain.",
        )
        spec.input(
            "discharged_hubbard_structure",
            valid_type=HubbardStructureData,
            required=False,
            help="A pre-converged discharged `HubbardStructureData` (U/V already self-consistent). If given, its SC-Hubbard run is skipped.",
        )
        spec.input(
            "charged_hubbard_structure",
            valid_type=HubbardStructureData,
            required=False,
            help="A pre-converged charged `HubbardStructureData` (U/V already self-consistent). If given, its SC-Hubbard run is skipped.",
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, work directories of all called calculation will be cleaned at the end of execution.",
        )
        spec.outline(
            cls.setup,
            cls.run_bulk_cation,
            if_(cls.should_run_hubbard)(
                cls.run_sc_hubbard,
                cls.inspect_sc_hubbard,
            ),
            cls.run_relax_unitcells,
            cls.build_supercells,
            cls.run_relax_SOC,
            cls.inspect_process,
            cls.results,
        )
        spec.exit_code(
            202,
            "ERROR_STRUCTURE_NOT_FOUND",
            message="The output relaxed structure of PwRelaxWorkChains not found.",
        )
        spec.exit_code(
            203,
            "ERROR_DFT_ENERGY_NOT_FOUND",
            message="The energy from the final scf calculation not found.",
        )
        spec.exit_code(
            204,
            "ERROR_MECHANICAL_UNSTABLE",
            message="The structure is not mechanically stable upon charging-discharging.",
        )
        spec.exit_code(
            205,
            "ERROR_CATION_NOT_FOUND",
            message="Could not infer or validate the cation.",
        )
        spec.exit_code(
            206,
            "ERROR_CATION_REFERENCE_NOT_FOUND",
            message="No bulk cation structure or fallback bulk cation DFT energy was provided.",
        )
        spec.exit_code(
            207,
            "ERROR_HUBBARD_SPEC_INVALID",
            message="The Hubbard spec is missing, malformed, references unknown kinds, or uses an unsupported spin configuration.",
        )
        spec.exit_code(
            208,
            "ERROR_CATION_IS_HUBBARD_ATOM",
            message="The cation is itself a Hubbard atom/neighbour, which is not allowed.",
        )
        spec.exit_code(
            209,
            "ERROR_SUB_PROCESS_FAILED_HUBBARD",
            message="A SelfConsistentHubbardWorkChain on a unitcell did not finish successfully.",
        )
        spec.output(
            "open_circuit_voltages",
            valid_type=orm.Dict,
            help="The dictionary containing the three voltages - average ocv and the ocv at high and low SOCs.",
        )
        spec.output(
            "common_workflow_output",
            valid_type=orm.Dict,
            help="The dictionary containing the voltages and all the structures relaxed within this workflow - charged/discharged unitcells and high/low SOC supercells.",
        )
        spec.output(
            "discharged_hubbard_structure",
            valid_type=HubbardStructureData,
            required=False,
            help="The converged discharged `HubbardStructureData` (relaxed geometry + self-consistent U/V). Only emitted in Hubbard mode.",
        )
        spec.output(
            "charged_hubbard_structure",
            valid_type=HubbardStructureData,
            required=False,
            help="The converged charged `HubbardStructureData` (relaxed geometry + self-consistent U/V). Only emitted in Hubbard mode.",
        )

    def setup(self):
        """
        Input validation and context setup.
        """
        # I store input ocv_parameters dictionary as context variable
        self.ctx.ocv_parameters_d = self.inputs.ocv_parameters.get_dict()
        try:
            cation = self.ctx.ocv_parameters_d.get("cation")
            if is_missing_cation(cation):
                cation = infer_cation_from_aiida_structure(self.inputs.structure)
                self.ctx.ocv_parameters_d["cation"] = cation
            self.ctx.cation = validate_cation(cation)
        except ValueError as exception:
            self.report(str(exception))
            return self.exit_codes.ERROR_CATION_NOT_FOUND

        self.ctx.ocv_parameters = orm.Dict(dict=self.ctx.ocv_parameters_d)

        # I store cation pseudo as context variable for putting it back after removing for structures without cations
        ocv_relax_inputs = AttributeDict(
            self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax")
        )
        try:
            self.ctx.cation_pseudo = ocv_relax_inputs.base.pw.pseudos[self.ctx.cation]
        except KeyError:
            self.report(
                f"The ocv_relax inputs do not contain a pseudo for cation {self.ctx.cation}."
            )
            return self.exit_codes.ERROR_CATION_NOT_FOUND
        # I store cell card as context variable for putting it back if supercells are vc-relaxed
        self.ctx.cell = ocv_relax_inputs.base.pw.parameters.get_dict()["CELL"]

        if ocv_relax_inputs.base.pw.parameters.get_dict()["SYSTEM"].get("starting_magnetization") is None:
            self.ctx.cation_magnetization = False
        else:
            self.ctx.cation_magnetization = True

        # Extended hubbard (DFT+U+V) mode is active if `hubbard_sc` namespace is supplied.
        self.ctx.hubbard_active = "hubbard_sc" in self.inputs
        if self.ctx.hubbard_active:
            exit_code = self._setup_hubbard()
            if exit_code is not None:
                return exit_code

    def _setup_hubbard(self):
        """Validate Hubbard inputs and stash the normalised spec. Returns an exit code on failure."""
        spec = self.ctx.ocv_parameters_d.get("hubbard")
        if spec is None and isinstance(self.inputs.structure, HubbardStructureData):
            try:
                spec = hub_func.extract_hubbard_spec(self.inputs.structure)
            except hub_func.HubbardSpecError as exception:
                self.report(str(exception))
                return self.exit_codes.ERROR_HUBBARD_SPEC_INVALID

        try:
            spec = hub_func.validate_hubbard_spec(
                spec, structure=self.inputs.structure, cation=self.ctx.cation
            )
        except hub_func.CationIsHubbardAtomError as exception:
            self.report(str(exception))
            return self.exit_codes.ERROR_CATION_IS_HUBBARD_ATOM
        except hub_func.HubbardSpecError as exception:
            self.report(str(exception))
            return self.exit_codes.ERROR_HUBBARD_SPEC_INVALID

        # hp.x supports only nspin in (1, 2): reject non-collinear / spin-orbit configurations.
        hubbard_sc_inputs = AttributeDict(self.exposed_inputs(SelfConsistentHubbardWorkChain, namespace="hubbard_sc"))
        system = hubbard_sc_inputs.scf.pw.parameters.get_dict().get("SYSTEM", {})

        if system.get("noncolin") or system.get("lspinorb") or system.get("nspin", 1) not in (1, 2):
            self.report("hp.x does not support non-collinear or spin-orbit calculations (nspin must be 1 or 2).")
            return self.exit_codes.ERROR_HUBBARD_SPEC_INVALID

        # If a restart unitcell was provided, in Hubbard mode it must carry Hubbard parameters.
        for key in ("discharged_unitcell_relaxed", "charged_unitcell_relaxed"):
            structure = self.inputs.get(key)
            if structure is not None and not isinstance(structure, HubbardStructureData):
                self.report(f"In Hubbard mode `{key}` must be a HubbardStructureData with converged U/V.")
                return self.exit_codes.ERROR_HUBBARD_SPEC_INVALID

        self.ctx.hubbard_spec = orm.Dict(dict=spec)
        return None

    def should_run_hubbard(self):
        """Return whether the self-consistent Hubbard step block should run."""
        return self.ctx.hubbard_active

    @staticmethod
    def _overrides_define_cation(overrides):
        """Return True if overrides explicitly define an ocv cation."""
        if not overrides:
            return False
        try:
            ocv_parameters = overrides.get("ocv_parameters", {})
        except AttributeError:
            return False
        return ocv_parameters is not None and "cation" in ocv_parameters

    @staticmethod
    def _validate_cation_reference(cation, ocv_parameters, has_bulk_cation_structure):
        """Require a cation reference energy when no bulk cation structure is supplied."""
        if has_bulk_cation_structure:
            return
        energy_key = f"DFT_energy_bulk_{cation}"
        if ocv_parameters.get(energy_key) is None:
            raise ValueError(
                f"Provide bulk_cation_structure or ocv_parameters['{energy_key}'] "
                f"for cation {cation}."
            )

    def _remove_cation_from_pw_inputs(self, pw_inputs):
        """Remove cation-specific pseudo and magnetisation entries from pw inputs.

        ``pw_inputs['parameters']`` is expected to already be a plain dict (callers convert it via
        ``get_dict()``). The magnetisation pop is guarded on the key being present so the same
        helper works for the plain `ocv_relax` inputs and for the `hubbard_sc` scf/relax inputs.
        """
        pw_inputs["pseudos"].pop(self.ctx.cation, None)
        magnetization = pw_inputs["parameters"].get("SYSTEM", {}).get("starting_magnetization")
        if isinstance(magnetization, dict):
            magnetization.pop(self.ctx.cation, None)

    def _adapt_pw_inputs_to_structure(self, pw_inputs, structure):
        """Re-key pseudos / starting_magnetization to ``structure`` kinds (Hubbard mode only).

        The `SelfConsistentHubbardWorkChain` may reorder or relabel kinds (hp.x puts Hubbard atoms
        first, and for U-only specs can split a kind into per-site types such as ``Mn0``/``Mn1``).
        For structures whose kind names are unchanged this reproduces the original mapping exactly,
        so it is a no-op for the common DFT+U+V case.
        """
        import re

        pseudos = pw_inputs["pseudos"]
        new_pseudos = {}
        for kind in structure.kinds:
            if kind.name in pseudos:
                new_pseudos[kind.name] = pseudos[kind.name]
                continue
            for key, pseudo in pseudos.items():
                if re.sub(r"\d", "", key) == kind.symbol:
                    new_pseudos[kind.name] = pseudo
                    break
        pw_inputs["pseudos"] = new_pseudos

        parameters = pw_inputs.get("parameters")
        magnetization = (
            parameters.get("SYSTEM", {}).get("starting_magnetization")
            if isinstance(parameters, dict)
            else None
        )
        if isinstance(magnetization, dict):
            new_magnetization = {}
            for kind in structure.kinds:
                if kind.name in magnetization:
                    new_magnetization[kind.name] = magnetization[kind.name]
                    continue
                for key, value in magnetization.items():
                    if re.sub(r"\d", "", key) == kind.symbol:
                        new_magnetization[kind.name] = value
                        break
            parameters["SYSTEM"]["starting_magnetization"] = new_magnetization

    @classmethod
    def get_protocol_filepath(cls):
        """
        Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols.
        """
        from importlib_resources import files
        from aiida_open_circuit_voltage.workflows import protocols as proto

        return files(proto) / "ocv.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        structure,
        protocol=None,
        overrides=None,
        bulk_cation_structure=None,
        discharged_unitcell_relaxed=None,
        charged_unitcell_relaxed=None,
        hp_code=None,
        **kwargs,
    ):
        """
        Return a builder prepopulated with inputs selected according to the chosen protocol.
        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use. It can be a ``HubbardStructureData``, in which case the Hubbard spec is inferred from it.
        :param bulk_cation_structure: the ``StructureData`` instance to get DFT energy of bulk cation.
        :param discharged_unitcell_relaxed: the ``StructureData`` instance that has been already relaxed.
        :param charged_unitcell_relaxed: the ``StructureData`` instance that has all the cations removed and has been relaxed.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol, usually takes the pseudo potential family and parallelisation options.
        :param hp_code: the ``Code`` instance configured for the ``quantumespresso.hp`` plugin. Required to run in DFT+U+V (Hubbard) mode, together with a Hubbard spec in ``ocv_parameters['hubbard']`` (or a ``HubbardStructureData`` as ``structure``).
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        if cls._overrides_define_cation(overrides):
            cation = inputs["ocv_parameters"].get("cation")
            if is_missing_cation(cation):
                cation = infer_cation_from_aiida_structure(structure)
            inputs["ocv_parameters"]["cation"] = validate_cation(cation)
        else:
            inputs["ocv_parameters"]["cation"] = infer_cation_from_aiida_structure(
                structure
            )

        cls._validate_cation_reference(
            inputs["ocv_parameters"]["cation"],
            inputs["ocv_parameters"],
            bulk_cation_structure is not None,
        )

        # Resolve the Hubbard spec (from ocv_parameters or an input HubbardStructureData) and build
        # the SelfConsistentHubbardWorkChain inputs. When no spec/hp_code is given this is a no-op
        # and the workchain runs as a plain-GGA OCV calculation.
        cation = inputs["ocv_parameters"]["cation"]
        hubbard_spec = cls._resolve_hubbard_spec(structure, inputs, hp_code, cation, kwargs)
        if hubbard_spec is not None:
            inputs["ocv_parameters"]["hubbard"] = hubbard_spec
            seed = hub_func.build_initialized_hubbard_structure(structure, hubbard_spec)
            # OCV, aiida-quantumespresso and aiida-hubbard share the same protocol names
            # (fast/balanced/stringent), so the protocol is forwarded as-is.
            hubbard_sc = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
                code,
                hp_code,
                seed,
                protocol=protocol or cls.get_default_protocol(),
                overrides=inputs.get("hubbard_sc"),
                **kwargs,
            )
            hubbard_sc.pop("hubbard_structure", None)
            hubbard_sc.pop("clean_workdir", None)

        args = (code, structure, protocol)
        ocv_relax = PwRelaxWorkChain.get_builder_from_protocol(
            *args, overrides=inputs["ocv_relax"], **kwargs
        )
        if bulk_cation_structure:
            args_cation = (code, bulk_cation_structure, protocol)
            scf = PwBaseWorkChain.get_builder_from_protocol(
                *args_cation, overrides=inputs.get("scf", None), **kwargs
            )
            scf["pw"].pop("structure", None)
            scf.pop("clean_workdir", None)

        ocv_relax.pop("structure", None)
        ocv_relax.pop("clean_workdir", None)
        ocv_relax.pop("parent_folder", None)

        builder = cls.get_builder()
        builder.ocv_relax = ocv_relax

        builder.structure = structure
        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.ocv_parameters = orm.Dict(dict=inputs["ocv_parameters"])

        if discharged_unitcell_relaxed:
            builder.discharged_unitcell_relaxed = discharged_unitcell_relaxed
        if charged_unitcell_relaxed:
            builder.charged_unitcell_relaxed = charged_unitcell_relaxed
        if bulk_cation_structure:
            builder.scf = scf
            builder.bulk_cation_structure = bulk_cation_structure
        else:
            builder.pop("scf")

        if hubbard_spec is not None:
            builder.hubbard_sc = hubbard_sc
        else:
            builder.pop("hubbard_sc", None)

        return builder

    @classmethod
    def _resolve_hubbard_spec(cls, structure, inputs, hp_code, cation, kwargs):
        """Return a validated Hubbard spec or ``None`` if the workchain should run plain GGA.

        The spec comes from ``ocv_parameters['hubbard']`` if present, otherwise it is extracted from
        ``structure`` when that is a ``HubbardStructureData`` carrying parameters. Raises ``ValueError``
        if a spec and ``hp_code`` are not supplied together or if the spin configuration is unsupported.
        """
        spec = inputs["ocv_parameters"].get("hubbard")

        # Without hp_code the workchain runs plain GGA (the default). An explicit spec without
        # hp_code is a mistake worth flagging; a HubbardStructureData on its own is not auto-promoted
        # to Hubbard mode, so passing one without hp_code still runs plain DFT.
        if hp_code is None:
            if spec is not None:
                raise ValueError("A Hubbard spec was set in ocv_parameters['hubbard'] but no hp_code was provided. Pass hp_code to run DFT+U+V, or remove the spec to run plain DFT.")
            return None

        if spec is None:
            if isinstance(structure, HubbardStructureData) and structure.hubbard.parameters:
                spec = hub_func.extract_hubbard_spec(structure)
            else:
                raise ValueError("hp_code was provided but no Hubbard spec was found. Set ocv_parameters['hubbard'] or pass a HubbardStructureData as `structure`.")

        spin_type = kwargs.get("spin_type")
        if spin_type in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
            raise ValueError("hp.x does not support non-collinear or spin-orbit calculations.")

        return hub_func.validate_hubbard_spec(spec, structure=structure, cation=cation)

    @classmethod
    def get_builder_from_json(cls, json_input, overrides=None):
        """
        Return a builder prepopulated with inputs selected from reading the provided json file.
        :param json_input: the path to a json file containing inputs, if provided all the inputs will be populated from this file
        :param overrides: optional dictionary of inputs to override the defaults of the protocol,
            if it is not provided it will be read from the .
        :return: a process builder instance with all inputs defined ready for launch.
        """
        with open(json_input) as json_file:
            data = json.load(json_file)

        inputs_j = data["inputs"]
        meta_j = data["meta"]

        if inputs_j.get("hubbard") or (overrides and overrides.get("ocv_parameters", {}).get("hubbard")):
            raise NotImplementedError("DFT+U+V (Hubbard) mode is not yet supported via get_builder_from_json; use get_builder_from_protocol with hp_code instead.")

        # loading structures
        structure = func.get_structuredata_from_optimade(inputs_j["structure"])
        structure_cation = func.get_structuredata_from_optimade(
            inputs_j["bulk_cation_structure"]
        )

        # loading parameters
        protocol = inputs_j["protocol"]
        # map the common-workflow "default" onto this plugin's default protocol name
        if protocol == "default":
            protocol = "balanced"
        code = inputs_j["engine"]["name"]

        # inputs are still populated from protocol but we replace these values with those read from json file
        if overrides is None:
            try:
                overrides = meta_j["overrides"]
            except KeyError:
                pass
        inputs = cls.get_protocol_inputs(protocol, overrides)

        # magnetic parameters
        magnetization_treatment = inputs_j["magnetization_treatment"]
        if magnetization_treatment == "collinear":
            spin_type = SpinType.COLLINEAR
        elif magnetization_treatment == "noncollinear":
            spin_type = SpinType.NON_COLLINEAR
        else:
            spin_type = SpinType.NONE
        spin_orbit = inputs_j["spin_orbit"]
        magnetization_per_site = inputs_j["magnetization_per_site"]
        # need to tell the builder that a list containing 0s means null initial magnetic moments
        if all(mag == 0 for mag in magnetization_per_site):
            initial_magnetic_moments = None
        else:
            initial_magnetic_moments = {kind.name: np.mean([magnetization for magnetization, sites in zip(magnetization_per_site, structure.sites) if sites.kind_name == kind.name]) for kind in structure.kinds}

        # other parameters
        cation = inputs_j.get("cation")
        if is_missing_cation(cation):
            cation = infer_cation_from_aiida_structure(structure)
        inputs["ocv_parameters"]["cation"] = validate_cation(cation)
        inputs["ocv_parameters"]["distance"] = inputs_j["supercell_distance"]
        inputs["ocv_parameters"]["volume_change_stability_threshold"] = inputs_j["volume_change_stability_threshold"]

        args = (code, structure, protocol)
        args_cation = (code, structure_cation, protocol)
        ocv_relax = PwRelaxWorkChain.get_builder_from_protocol(*args, overrides=inputs["ocv_relax"], spin_type=spin_type, initial_magnetic_moments=initial_magnetic_moments,)
        scf = PwBaseWorkChain.get_builder_from_protocol(*args_cation, overrides=inputs.get("scf", None))

        # loading k-points
        kpoints_distance = inputs_j["kpoints_distance"]
        if kpoints_distance:
            ocv_relax["base"]["kpoints_distance"] = orm.Float(kpoints_distance)
            ocv_relax["base_final_scf"]["kpoints_distance"] = orm.Float(kpoints_distance)
            scf["kpoints_distance"] = orm.Float(kpoints_distance)
        else:
            kpoints_mesh = inputs_j["kpoints_mesh"]

        # Specifying spin-orbit here as it doesn't exist in aiida-quantumespresso
        if spin_orbit:
            ocv_relax.base["pw"]["parameters"]["SYSTEM"]["lspinorb"] = spin_orbit
            ocv_relax.base_final_scf["pw"]["parameters"]["SYSTEM"]["lspinorb"] = spin_orbit

        ocv_relax.pop("structure", None)
        ocv_relax.pop("clean_workdir", None)
        ocv_relax.pop("parent_folder", None)
        scf["pw"].pop("structure", None)
        scf.pop("clean_workdir", None)

        builder = cls.get_builder()
        builder.ocv_relax = ocv_relax
        builder.scf = scf
        builder.structure = structure
        builder.bulk_cation_structure = structure_cation

        builder.clean_workdir = orm.Bool(inputs["clean_workdir"])
        builder.ocv_parameters = orm.Dict(dict=inputs["ocv_parameters"])

        return builder

    def run_bulk_cation(self):
        """
        Runs a PwBaseWorkChain to calculate DFT energy of bulk cation structure, if that structure is provided.
        Otherwise the energy is read from inputs.
        """
        if self.inputs.get("bulk_cation_structure"):

            bulk_cation_structure = self.inputs.bulk_cation_structure

            self.report(f"Bulk cation structure <{bulk_cation_structure.pk}> provided, I will use this structure to calculate scf energy of {self.ctx.cation}.")
            qb = orm.QueryBuilder()
            qb.append(orm.StructureData, filters={"uuid": {"==": bulk_cation_structure.uuid}}, tag="struct", )
            qb.append(WorkflowFactory("quantumespresso.pw.base"), with_incoming="struct", tag="base", 
                      filters={"and": [{"attributes.process_state": {"==": "finished"}}, {"attributes.exit_status": {"==": 0}},]},)

            if qb.count():
                wc = qb.all(flat=True)[-1]
                self.report(f"Workchain <{wc.pk}> corresponding to bulk cation found")
                return ToContext(cation_workchain=append_(wc))

            else:
                inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="scf"))
                inputs.pw.structure = bulk_cation_structure

                inputs.metadata.call_link_label = "bulk_cation_scf"
                inputs.metadata.label = "bulk_cation_scf"
                inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

                running = self.submit(PwBaseWorkChain, **inputs)

                self.report(f"launching PwBaseWorkChain <{running.pk}> on bulk cation structure")
                return ToContext(cation_workchain=append_(running))
        else:
            energy_key = f"DFT_energy_bulk_{self.ctx.cation}"
            if self.ctx.ocv_parameters_d.get(energy_key) is None:
                self.report(f"Bulk cation structure not provided and ocv_parameters['{energy_key}'] is missing.")
                return self.exit_codes.ERROR_CATION_REFERENCE_NOT_FOUND
            self.report(f"Bulk cation structure not provided, so I will use the input scf energy of {self.ctx.cation}.")
            # I put this context dictionary as none so that the energy can be read from ocv_relax_parameters dictionary
            self.ctx.bulk_cation_d = None

    def _prepare_hubbard_sc_inputs(self, hubbard_structure, label, remove_cation=False):
        """Prepare inputs for one SelfConsistentHubbardWorkChain run."""
        inputs = AttributeDict(self.exposed_inputs(SelfConsistentHubbardWorkChain, namespace="hubbard_sc"))
        inputs.hubbard_structure = hubbard_structure
        # Forward the OCV `clean_workdir` (default False) so the SC loop keeps its restart folders;
        # the SC workchain otherwise defaults clean_workdir to True.
        inputs.clean_workdir = self.inputs.clean_workdir

        if remove_cation:
            # The charged unitcell has no cation, so drop the cation pseudo / magnetisation from the
            # scf and relax namespaces of the SC workchain.
            inputs.scf.pw.parameters = inputs.scf.pw.parameters.get_dict()
            self._remove_cation_from_pw_inputs(inputs.scf.pw)
            inputs.relax.base.pw.parameters = inputs.relax.base.pw.parameters.get_dict()
            self._remove_cation_from_pw_inputs(inputs.relax.base.pw)

        inputs.metadata.call_link_label = label
        inputs.metadata.label = label
        return prepare_process_inputs(SelfConsistentHubbardWorkChain, inputs)

    def run_sc_hubbard(self):
        """Launch self-consistent DFT+U+V on the discharged and charged unitcells in parallel.

        A side is skipped when either a pre-converged `*_hubbard_structure` (skips only the SC loop)
        or a relaxed `*_unitcell_relaxed` (skips the SC loop and the subsequent relax) is provided.
        The converged-U/V structure used to seed the fixed-U/V relax is stored in
        ``ctx.{discharged,charged}_hubbard_structure``.
        """
        self.ctx.launched_discharged_hubbard = False
        self.ctx.launched_charged_hubbard = False

        # Discharged unitcell.
        if self.inputs.get("discharged_hubbard_structure") is not None:
            self.ctx.discharged_hubbard_structure = self.inputs.discharged_hubbard_structure
            self.report("Using provided converged discharged HubbardStructureData; skipping its SC-Hubbard run.")
        elif self.inputs.get("discharged_unitcell_relaxed") is not None:
            self.report("Relaxed discharged unitcell provided; skipping its SC-Hubbard run.")
        else:
            seed = hub_func.initialize_hubbard_structure(self.inputs.structure, self.ctx.hubbard_spec)["hubbard_structure"]
            inputs = self._prepare_hubbard_sc_inputs(seed, "discharged_hubbard_sc")
            running = self.submit(SelfConsistentHubbardWorkChain, **inputs)
            self.report(f"launching SelfConsistentHubbardWorkChain <{running.pk}> on discharged unitcell")
            self.ctx.launched_discharged_hubbard = True
            self.to_context(discharged_hubbard_workchain=append_(running))

        # Charged unitcell.
        if self.inputs.get("charged_hubbard_structure") is not None:
            self.ctx.charged_hubbard_structure = self.inputs.charged_hubbard_structure
            self.report("Using provided converged charged HubbardStructureData; skipping its SC-Hubbard run.")
        elif self.inputs.get("charged_unitcell_relaxed") is not None:
            self.report("Relaxed charged unitcell provided; skipping its SC-Hubbard run.")
        else:
            charged = func.get_charged(self.inputs.structure, orm.Str(self.ctx.cation))["decationised_structure"]
            seed = hub_func.initialize_hubbard_structure(charged, self.ctx.hubbard_spec)["hubbard_structure"]
            inputs = self._prepare_hubbard_sc_inputs(seed, "charged_hubbard_sc", remove_cation=True)
            running = self.submit(SelfConsistentHubbardWorkChain, **inputs)
            self.report(f"launching SelfConsistentHubbardWorkChain <{running.pk}> on charged unitcell")
            self.ctx.launched_charged_hubbard = True
            self.to_context(charged_hubbard_workchain=append_(running))

    def inspect_sc_hubbard(self):
        """Collect the converged Hubbard unitcells; fail if a launched SC-Hubbard run did not converge."""
        if self.ctx.launched_discharged_hubbard:
            workchain = self.ctx.discharged_hubbard_workchain[-1]
            if not workchain.is_finished_ok:
                self.report(f"discharged SelfConsistentHubbardWorkChain failed with exit status {workchain.exit_status}")
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HUBBARD
            self.ctx.discharged_hubbard_structure = workchain.outputs.hubbard_structure

        if self.ctx.launched_charged_hubbard:
            workchain = self.ctx.charged_hubbard_workchain[-1]
            if not workchain.is_finished_ok:
                self.report(f"charged SelfConsistentHubbardWorkChain failed with exit status {workchain.exit_status}")
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HUBBARD
            self.ctx.charged_hubbard_structure = workchain.outputs.hubbard_structure

        if hasattr(self.ctx, "discharged_hubbard_structure"):
            self.out("discharged_hubbard_structure", self.ctx.discharged_hubbard_structure)
            # Warn if hp.x relabelled kinds (U-only specs): downstream transfers fall back to averaging.
            original_kinds = {kind.name for kind in self.inputs.structure.kinds}
            converged_kinds = {kind.name for kind in self.ctx.discharged_hubbard_structure.kinds}
            if not converged_kinds.issubset(original_kinds):
                self.report("hp.x relabelled kinds during the SC-Hubbard run; per-symbol averaging will be used where exact parameter transfer is not possible.")
        if hasattr(self.ctx, "charged_hubbard_structure"):
            self.out("charged_hubbard_structure", self.ctx.charged_hubbard_structure)

    def run_relax_unitcells(self):
        """Launch discharged and charged unitcell calculations before waiting."""
        # Saving the bulk cation DFT energy as context variable
        if self.inputs.get("bulk_cation_structure"):
            try:
                self.ctx.bulk_cation_d = self.ctx.cation_workchain[-1].outputs.output_parameters
            except exceptions.NotExistent:
                self.report("The PwBaseWorkChain did not generate output parameters for bulk cation structure")
                return self.exit_codes.ERROR_DFT_ENERGY_NOT_FOUND

        discharged_workchain = self._launch_discharged_unitcell()
        charged_workchain = self._launch_charged_unitcell()

        if discharged_workchain is not None:
            self.to_context(discharged_workchain=append_(discharged_workchain))
        if charged_workchain is not None:
            self.to_context(charged_workchain=append_(charged_workchain))

    def _launch_discharged_unitcell(self):
        """Submit or reuse the discharged unitcell calculation."""
        # If relaxed unitcell is provided, I run a PwBaseWorkChain on that structure
        if self.inputs.get("discharged_unitcell_relaxed"):
            # I store the input relaxed discharged unitcell as context variable
            self.ctx.discharged_unitcell_relaxed = (self.inputs.discharged_unitcell_relaxed)
            self.report(f"Relaxed discharged unitcell <{self.ctx.discharged_unitcell_relaxed.pk}> already provided")

            qb = orm.QueryBuilder()
            qb.append(orm.StructureData, filters={"uuid": {"==": self.ctx.discharged_unitcell_relaxed.uuid}}, tag="struct",)
            qb.append(WorkflowFactory("quantumespresso.pw.relax"), with_outgoing="struct", tag="base", 
                      filters={"and": [{"attributes.process_state": {"==": "finished"}}, {"attributes.exit_status": {"==": 0}},]},)

            if qb.count():
                wc = qb.all(flat=True)[-1]
                self.report(f"Workchain <{wc.pk}> corresponding to relaxed discharged unitcell found")
                return wc

            else:
                inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax"))["base_final_scf"]
                inputs.pw.structure = self.ctx.discharged_unitcell_relaxed
                if self.ctx.hubbard_active:
                    inputs.pw.parameters = inputs.pw.parameters.get_dict()
                    self._adapt_pw_inputs_to_structure(inputs.pw, self.ctx.discharged_unitcell_relaxed)
                inputs.metadata.call_link_label = "discharged_scf"
                inputs.metadata.label = "discharged_scf"

                inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

                running = self.submit(PwBaseWorkChain, **inputs)
                self.report(f"launching PwBaseWorkChain <{running.pk}> on relaxed discharged structure")

                return running

        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax"))

        # In Hubbard mode the discharged unitcell is the converged HubbardStructureData (fixed-U/V
        # vc-relax); otherwise it is the plain input structure.
        if self.ctx.hubbard_active:
            discharged_unitcell = self.ctx.discharged_hubbard_structure
        else:
            discharged_unitcell = self.inputs.structure
        discharged_unitcell.set_extra("relaxed", False)
        discharged_unitcell.set_extra("supercell", False)

        inputs["structure"] = discharged_unitcell

        if self.ctx.hubbard_active:
            inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
            inputs.base_final_scf.pw.parameters = (inputs.base_final_scf.pw.parameters.get_dict())
            self._adapt_pw_inputs_to_structure(inputs.base.pw, discharged_unitcell)
            self._adapt_pw_inputs_to_structure(inputs.base_final_scf.pw, discharged_unitcell)

        inputs.metadata.call_link_label = "discharged_relax"
        inputs.metadata.label = "discharged_relax"

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f"launching PwRelaxWorkChain <{running.pk}> on discharged structure")
        return running

    def _launch_charged_unitcell(self):
        """Submit or reuse the charged unitcell calculation."""
        # If relaxed unitcell is provided, I run a PwBaseWorkChain on that structure
        if self.inputs.get("charged_unitcell_relaxed"):
            # I store the input relaxed discharged unitcell as context variable
            self.ctx.charged_unitcell_relaxed = self.inputs.charged_unitcell_relaxed
            self.report(f"Relaxed charged unitcell <{self.ctx.charged_unitcell_relaxed.pk}> already provided.")

            qb = orm.QueryBuilder()
            qb.append(orm.StructureData, filters={"uuid": {"==": self.ctx.charged_unitcell_relaxed.uuid}}, tag="struct",)
            qb.append(WorkflowFactory("quantumespresso.pw.relax"), with_outgoing="struct", tag="base", 
                      filters={"and": [{"attributes.process_state": {"==": "finished"}}, {"attributes.exit_status": {"==": 0}},]},)

            if qb.count():
                wc = qb.all(flat=True)[-1]
                self.report(f"Workchain <{wc.pk}> corresponding to relaxed charged unitcell found")
                return wc

            else:
                inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax"))["base_final_scf"]
                inputs.pw.structure = self.ctx.charged_unitcell_relaxed

                # Since it's in orm.Dict datatype, I need to get the python dict to make changes to it
                inputs.pw.parameters = inputs.pw.parameters.get_dict()
                if self.ctx.hubbard_active:
                    self._adapt_pw_inputs_to_structure(inputs.pw, self.ctx.charged_unitcell_relaxed)
                else:
                    self._remove_cation_from_pw_inputs(inputs.pw)
                inputs.metadata.call_link_label = "charged_scf"
                inputs.metadata.label = "charged_scf"

                inputs = prepare_process_inputs(PwBaseWorkChain, inputs)

                running = self.submit(PwBaseWorkChain, **inputs)
                self.report(f"launching PwBaseWorkChain <{running.pk}> on relaxed charged structure")

                return running

        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax"))

        # In Hubbard mode the charged unitcell is the converged HubbardStructureData (already
        # cation-free); otherwise build it by removing all cations from the input structure.
        if self.ctx.hubbard_active:
            charged_unitcell = self.ctx.charged_hubbard_structure
        else:
            charged_unitcell = func.get_charged(self.inputs.structure, orm.Str(self.ctx.cation))["decationised_structure"]
        charged_unitcell.set_extra("relaxed", False)
        charged_unitcell.set_extra("supercell", False)

        inputs["structure"] = charged_unitcell

        ## Since it's in orm.Dict datatype, I need to get the python dict to make changes to it
        inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
        inputs.base_final_scf.pw.parameters = inputs.base_final_scf.pw.parameters.get_dict()

        if self.ctx.hubbard_active:
            # Re-key pseudos / magnetisation to the cation-free Hubbard kinds (the cation is
            # automatically dropped since it is absent from the structure).
            self._adapt_pw_inputs_to_structure(inputs.base.pw, charged_unitcell)
            self._adapt_pw_inputs_to_structure(inputs.base_final_scf.pw, charged_unitcell)
        else:
            # Removing cation pseudopotential since this structure no longer has any cation in it
            self._remove_cation_from_pw_inputs(inputs.base.pw)
            self._remove_cation_from_pw_inputs(inputs.base_final_scf.pw)

        inputs.metadata.call_link_label = "charged_relax"
        inputs.metadata.label = "charged_relax"

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f"launching PwRelaxWorkChain <{running.pk}> on charged structure")
        return running

    def build_supercells(self):
        """
        Making all types of supercells here.
        """
        # Saving the relaxed structures in context variables
        try:
            self.ctx.discharged_unitcell_relaxed
        except AttributeError:
            try:
                self.ctx.discharged_unitcell_relaxed = self.ctx.discharged_workchain[-1].outputs.output_structure
            except exceptions.NotExistent:
                self.report("The PwRelaxWorkChains did not generate output structures of discharged unitcell")
                return self.exit_codes.ERROR_STRUCTURE_NOT_FOUND

        try:
            self.ctx.charged_unitcell_relaxed
        except AttributeError:
            try:
                self.ctx.charged_unitcell_relaxed = self.ctx.charged_workchain[-1].outputs.output_structure
            except exceptions.NotExistent:
                self.report("The PwRelaxWorkChains did not generate output structures of charged unitcell")
                return self.exit_codes.ERROR_STRUCTURE_NOT_FOUND

        self.ctx.discharged_unitcell_relaxed.set_extra("relaxed", True)
        self.ctx.discharged_unitcell_relaxed.set_extra("supercell", False)
        self.ctx.charged_unitcell_relaxed.set_extra("relaxed", True)
        self.ctx.charged_unitcell_relaxed.set_extra("supercell", False)

        volume_charged = self.ctx.charged_unitcell_relaxed.get_cell_volume()
        volume_discharged = self.ctx.discharged_unitcell_relaxed.get_cell_volume()
        volume_change = (volume_charged - volume_discharged) / volume_discharged

        # I check mechanical stability on cation removal
        if self.ctx.ocv_parameters_d["volume_change_stability"]:
            threshold = self.ctx.ocv_parameters_d["volume_change_stability_threshold"]
            if abs(volume_change) > threshold:
                self.report(f"The Volume changed <{volume_change}> too much upon cation removal")
                return self.exit_codes.ERROR_MECHANICAL_UNSTABLE
            else:
                self.report(f"Volume change <{volume_change}> is within the threshold <{threshold}>")

        # I make the constrained unitcell and store it as context variable
        self.ctx.constrained_unitcell = func.get_constrained_charged(
            self.ctx.discharged_unitcell_relaxed,
            orm.Str(self.ctx.cation),
            orm.Float(volume_charged),
        )

        # I make the supercells with same number of non cationic species. In Hubbard mode the
        # supercell lattice must not be standardised (rotated), so the converged unitcell U/V can be
        # mapped onto it by position via HubbardUtils.get_hubbard_for_supercell.
        discharged_supercell_relaxed = func.make_supercell(
            self.ctx.discharged_unitcell_relaxed,
            self.ctx.ocv_parameters_d["distance"],
            standardize=not self.ctx.hubbard_active,
        )
        discharged_supercell_relaxed.set_extra("relaxed", True)
        discharged_supercell_relaxed.set_extra("supercell", True)

        res = func.get_unique_cation_sites(discharged_supercell_relaxed, orm.Str(self.ctx.cation))
        all_cation_indices, unique_cation_indices = (res["all_cation_indices"], res["unique_cation_indices"],)

        # I make the low and high SOC supercells and store the dictionray of structures as context variables
        self.ctx.low_SOC_supercells_d = func.get_low_SOC(discharged_supercell_relaxed, unique_cation_indices)
        # the new volume of the high_SOC supercell, based on the scaling factor i.e. the volume ratio
        scaling_factor = volume_charged / volume_discharged
        new_volume = scaling_factor * discharged_supercell_relaxed.get_cell_volume()
        # I scale the discharged supercell wrt volume ratio to get a supercell with the same proportional volume as the charged unitcell
        self.ctx.high_SOC_supercells_d = func.get_high_SOC(
            discharged_supercell_relaxed,
            orm.Float(new_volume),
            all_cation_indices,
            unique_cation_indices,
        )

        # The SOC supercells only need converged U/V if at least one SOC OCV is requested; for an
        # average-only run we skip the transfer entirely (those supercells are never relaxed).
        if self.ctx.hubbard_active and (self.ctx.ocv_parameters_d["do_low_SOC_OCV"] or self.ctx.ocv_parameters_d["do_high_SOC_OCV"]):
            self._attach_hubbard_to_supercells()

        return

    def _attach_hubbard_to_supercells(self):
        """Re-attach converged U/V parameters onto the plain SOC structures (Hubbard mode).

        Low-SOC supercells get the exact discharged-unitcell parameters (same composition); the
        constrained-charged unitcell and high-SOC supercells get per-symbol-averaged charged-unitcell
        parameters (those cells are scaled discharged supercells, not commensurate with the relaxed
        charged unitcell, so exact mapping is impossible). The constrained and high-SOC cells share
        the same averaged values, keeping the high-SOC OCV formula self-consistent.
        """
        spec = self.ctx.hubbard_spec

        self.ctx.constrained_unitcell = hub_func.get_hubbard_averaged(self.ctx.constrained_unitcell, self.ctx.charged_unitcell_relaxed, spec)["hubbard_structure"]

        self.ctx.low_SOC_supercells_d = {
            key: hub_func.get_hubbard_supercell(structure, 
            self.ctx.discharged_unitcell_relaxed, spec)["hubbard_structure"]
            for key, structure in self.ctx.low_SOC_supercells_d.items()
        }

        self.ctx.high_SOC_supercells_d = {
            key: hub_func.get_hubbard_averaged(structure, 
            self.ctx.charged_unitcell_relaxed, spec)["hubbard_structure"]
            for key, structure in self.ctx.high_SOC_supercells_d.items()
        }

    def _prepare_soc_relax_inputs(self, structure, label, remove_cation=False):
        """Prepare SOC relaxation inputs for one structure."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace="ocv_relax"))
        inputs["structure"] = structure

        # Since it's in orm.Dict datatype, I need to get the python dict to make changes to it.
        inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
        if remove_cation:
            inputs.base_final_scf.pw.parameters = (inputs.base_final_scf.pw.parameters.get_dict())

        if not self.ctx.ocv_parameters_d["SOC_vc_relax"]:
            inputs.base.pw.parameters["CONTROL"]["calculation"] = "relax"
            inputs.base.pw.parameters.pop("CELL", None)

        if remove_cation:
            self._remove_cation_from_pw_inputs(inputs.base.pw)
            self._remove_cation_from_pw_inputs(inputs.base_final_scf.pw)

        inputs.metadata.call_link_label = label
        inputs.metadata.label = label

        return prepare_process_inputs(PwRelaxWorkChain, inputs)

    def _submit_soc_relax(self, structure, label, context_key, remove_cation=False):
        """Submit one SOC relaxation and append it to the requested context key."""
        inputs = self._prepare_soc_relax_inputs(structure, label, remove_cation=remove_cation)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f"launching PwRelaxWorkChain <{running.pk}> on {label} structure")
        self.to_context(**{context_key: append_(running)})

    def run_relax_SOC(self):
        """
        Launch enabled low-SOC, constrained charged, and high-SOC relaxations.

        If SOC_relax_all_supercells is False, only structure 00 is launched for
        each SOC limit. If True, all generated structures are launched and the
        lowest-energy output is selected in inspect_process.
        """
        relax_all = self.ctx.ocv_parameters_d["SOC_relax_all_supercells"]

        if self.ctx.ocv_parameters_d["do_low_SOC_OCV"]:
            if relax_all:
                low_SOC_structures = self.ctx.low_SOC_supercells_d.items()
            else:
                low_SOC_structures = [
                    (
                        "low_SOC_structure_00",
                        self.ctx.low_SOC_supercells_d["low_SOC_structure_00"],
                    )
                ]

            for idx, (key, low_structure) in enumerate(low_SOC_structures):
                label = f"low_SOC_{idx:02d}_relax" if relax_all else "low_SOC_relax"
                self._submit_soc_relax(low_structure, label, "low_SOC_workchains")
        else:
            self.report("I do not perform low SOC calculations.")

        if self.ctx.ocv_parameters_d["do_high_SOC_OCV"]:
            self._submit_soc_relax(
                self.ctx.constrained_unitcell,
                "constrained_charged_relax",
                "constrained_charged_workchain",
                remove_cation=True,
            )

            if relax_all:
                high_SOC_structures = self.ctx.high_SOC_supercells_d.items()
            else:
                high_SOC_structures = [
                    (
                        "high_SOC_structure_00",
                        self.ctx.high_SOC_supercells_d["high_SOC_structure_00"],
                    )
                ]

            for idx, (key, high_structure) in enumerate(high_SOC_structures):
                label = f"high_SOC_{idx:02d}_relax" if relax_all else "high_SOC_relax"
                self._submit_soc_relax(high_structure, label, "high_SOC_workchains")
        else:
            self.report("I do not perform constrained charged or high SOC calculations.")

    def inspect_process(self):
        """
        Inspects the workchains to see if all the required energies are properly calculated at various states of charge.
        """
        try:
            if self.ctx.ocv_parameters_d["do_low_SOC_OCV"]:
                # Select the lowest-energy output among the low-SOC structures that were launched.
                low_SOC_dicts = {
                    f"low_SOC_outputs_{idx:02d}": workchain.outputs.output_parameters
                    for idx, workchain in enumerate(self.ctx.low_SOC_workchains)
                }
                self.ctx.low_SOC_dict = func.get_lowest_energy(**low_SOC_dicts)
                self.ctx.low_SOC_supercells_relaxed = {
                    f"low_SOC_structure_{idx:02d}": workchain.outputs.output_structure
                    for idx, workchain in enumerate(self.ctx.low_SOC_workchains)
                }
            else:
                self.ctx.low_SOC_dict = None
                self.ctx.low_SOC_supercells_relaxed = None
            if self.ctx.ocv_parameters_d["do_high_SOC_OCV"]:
                # Select the lowest-energy output among the high-SOC structures that were launched.
                high_SOC_dicts = {
                    f"high_SOC_outputs_{idx:02d}": workchain.outputs.output_parameters
                    for idx, workchain in enumerate(self.ctx.high_SOC_workchains)
                }
                self.ctx.high_SOC_dict = func.get_lowest_energy(**high_SOC_dicts)
                self.ctx.high_SOC_supercells_relaxed = {
                    f"high_SOC_structure_{idx:02d}": workchain.outputs.output_structure
                    for idx, workchain in enumerate(self.ctx.high_SOC_workchains)
                }
                self.ctx.constrained_charged_d = self.ctx.constrained_charged_workchain[
                    -1
                ].outputs.output_parameters
                self.ctx.constrained_charged_relaxed = (
                    self.ctx.constrained_charged_workchain[-1].outputs.output_structure
                )
            else:
                self.ctx.high_SOC_dict = None
                self.ctx.high_SOC_supercells_relaxed = None
                self.ctx.constrained_charged_d = None
                self.ctx.constrained_charged_relaxed = None
        except exceptions.NotExistent:
            self.report(
                "the high/low SOC PwRelaxWorkChains did not generate output parameters/structures"
            )
            return self.exit_codes.ERROR_DFT_ENERGY_NOT_FOUND
        try:
            self.ctx.charged_d = self.ctx.charged_workchain[
                -1
            ].outputs.output_parameters
            self.ctx.discharged_d = self.ctx.discharged_workchain[
                -1
            ].outputs.output_parameters
        except AttributeError:
            self.report(
                "the charged/discharged PwBaseWorkChains did not generate output parameters for charged and discharged structures"
            )
            return self.exit_codes.ERROR_DFT_ENERGY_NOT_FOUND

    def results(self):
        """
        Returns the OCVs at various states of charge and outputs the dictionary based on the common workflow standards.
        """
        ocv = func.get_OCVs(
            self.ctx.ocv_parameters,
            self.ctx.discharged_d,
            self.ctx.charged_d,
            self.ctx.bulk_cation_d,
            self.ctx.constrained_charged_d,
            self.ctx.low_SOC_dict,
            self.ctx.high_SOC_dict,
        )
        self.report(f"Open circuit voltages calculated and outputed in <{ocv.id}>")
        self.out("open_circuit_voltages", ocv)

        if self.ctx.low_SOC_supercells_relaxed:
            low_SOC_structures = func.get_optimade_structures(
                **self.ctx.low_SOC_supercells_relaxed
            )
        else:
            low_SOC_structures = None

        if self.ctx.high_SOC_supercells_relaxed:
            high_SOC_structures = func.get_optimade_structures(
                **self.ctx.high_SOC_supercells_relaxed
            )
        else:
            high_SOC_structures = None

        json_out = func.get_json_outputs(
            ocv,
            self.ctx.discharged_unitcell_relaxed,
            self.ctx.charged_unitcell_relaxed,
            self.ctx.constrained_charged_relaxed,
            low_SOC_structures,
            high_SOC_structures,
        )
        self.out("common_workflow_output", json_out)
