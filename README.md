# aiida-open_circuit_voltage
An AiiDA plugin to calcutlate open circuit voltages at various charge of states for any arbitrary cathode material.

## Requirements
AiiDA see https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html

Supercellor see https://github.com/lekah/supercellor

## Installation
To install from the sources run:
```
git clone https://github.com/tsthakur/aiida-open_circuit_voltage.git

pip install aiida-open_circuit_voltage
```

After this run:
```
reentry scan 
```
This command ensures that aiida correctly recognises the plugin's newly added entry points.

## Example Run
A jupyter notebook along with an AiiDA compatible structure (olivine LiFePO4) is bundled as an example to run the workchain. 
To import the structure refer to the instructions here - https://aiida.readthedocs.io/projects/aiida-core/en/latest/howto/share_data.html#importing-an-archive
