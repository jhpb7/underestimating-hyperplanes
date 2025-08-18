from setuptools import find_packages, setup

setup(
    name="underestimating_hyperplanes",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.7",
)
