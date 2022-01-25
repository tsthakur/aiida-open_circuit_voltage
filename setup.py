import json
from pathlib import Path
from setuptools import setup, find_packages

with Path('README.md').open('r', encoding='utf8') as handle:
    LONG_DESCRIPTION = handle.read()

if __name__ == '__main__':
    with Path('setup.json').open('r', encoding='utf8') as info:
        kwargs = json.load(info)
    setup(
        packages=find_packages(
            include=['aiida_open_circuit_voltage', 'aiida_open_circuit_voltage.*']),
        package_data={
            '': ['*'],
        },
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        **kwargs)