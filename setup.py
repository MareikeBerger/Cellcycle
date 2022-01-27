import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cellcycle", # name of my package
    version="0.0.1",
    author="Mareike Berger",
    author_email="m.berger@amolf.nl",
    description="A package to run simulations of the cell cycle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "matplotlib", "pandas", "seaborn", "treelib", "scipy", "h5py", "gitpython", "tables", "cmocean", "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points= {'console_scripts': ['cellcycle_run=cellcycle.mainClass:run']}
)
