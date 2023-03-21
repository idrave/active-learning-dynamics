from setuptools import setup, find_packages

required = [
    'numpy >= 1.18',
    'opencv-python >= 4.2',
    'netaddr >= 0.8',
    'netifaces >= 0.10',
    'myqr >= 2.3',
    'robomaster >= 0.1.1.68',
    'gymnasium >= 0.27.1',
    'scikit-learn >= 1.0.2',
    'scipy >= 1.7.3'
] # TODO: check if anything is missing

extras = {}
setup(
    name='alrd',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.6.5',
    include_package_data=True,
    install_requires=required,
    extras_require=extras
    )