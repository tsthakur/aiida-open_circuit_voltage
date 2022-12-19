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


## Acknowledgements

This project has received funding from the European Union’s [Horizon 2020 research and innovation programme](https://ec.europa.eu/programmes/horizon2020/en) under grant agreement [No 957189](https://cordis.europa.eu/project/id/957189). The project is part of BATTERY 2030+, the large-scale European research initiative for inventing the sustainable batteries of the future.
