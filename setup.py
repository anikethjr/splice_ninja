import setuptools

setuptools.setup(
    name='splice_ninja',
    version='0.1',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "argparse", "wandb", "h5py", "tqdm", \
                      "scipy", "scikit-learn", "matplotlib", "seaborn",\
                      "torch", "lightning", \
                      "pyfaidx", "joblib", \
                      "biopython", "einsum", "rotary-embedding-torch"]
)