# aiida-open_circuit_voltage
An AiiDA plugin to calculate open circuit voltages at various charge of states for any arbitrary cathode material.

## Requirements
- AiiDA (`aiida-core` 2.8), see https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html
- `aiida-quantumespresso` 5.x, see https://github.com/aiidateam/aiida-quantumespresso
- `supercellor`, see https://github.com/lekah/supercellor

## Installation
To install from the sources run:
```
git clone https://github.com/tsthakur/aiida-open_circuit_voltage.git
cd aiida-open_circuit_voltage
pip install -e .
```

## Example Run
A jupyter notebook along with an AiiDA compatible structure (olivine LiFePO4) is bundled as an example to run the workchain. 
To import the structure refer to the instructions here - https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/share_data.html#importing-an-archive

## DFT+U+V (extended Hubbard)

**Temporarily unsupported with aiida-quantumespresso 5.x.** 
`aiida-hubbard` is currently not compatible with aiida-quantumespresso 5 (refer to https://github.com/aiidateam/aiida-hubbard/pull/119).
To run DFT+U+V use this plugin at git tag `v0.6` (`git checkout v0.6`) together with aiida-quantumespresso 4.17 and aiida-hubbard 0.5.

### How it works (v0.6)

By default the workchain computes all energies at plain GGA (PBEsol). 
To instead use self-consistent extended Hubbard parameters (`hp.x` linear response), pass an `hp_code` and a Hubbard *spec* describing which manifolds to correct. 
The discharged and charged unitcells are run through [`aiida-hubbard`](https://github.com/aiidateam/aiida-hubbard)'s `SelfConsistentHubbardWorkChain`; the converged U/V are then used (exactly for low-SOC supercells, per-symbol-averaged for the charged-composition cells) in fixed-U/V relaxations for every voltage-related energy.

## Acknowledgements

This project has received funding from the European Union’s [Horizon 2020 research and innovation programme](https://ec.europa.eu/programmes/horizon2020/en) under grant agreement [No 957189](https://cordis.europa.eu/project/id/957189). The project is part of BATTERY 2030+, the large-scale European research initiative for inventing the sustainable batteries of the future.
