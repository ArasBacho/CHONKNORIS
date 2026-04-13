import setuptools

packages = [
    'chonknoris',
]

install_requires = [
    'numpy',
    'scipy',
    'torch',
    'gpytorch',
    'lightning',
    'linear_operator',
    'pandas',
    'matplotlib',
    'neuraloperator',
    'pykolesky',
]

setuptools.setup(
    name="chonknoris",
    version="1.0",
    author="Aras Bacho, Aleksei G. Sorokin, Xianjin Yang, Théo Bourdais, "
           "Edoardo Calvello, Matthieu Darcy, Alexander Hsu, Bamdad Hosseini, Houman Owhadi",
    description="CHONKNORIS: Operator Learning at Machine Precision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArasBacho/CHONKNORIS",
    packages=packages,
    install_requires=install_requires,
    python_requires=">=3.10",
    include_package_data=True,
)
