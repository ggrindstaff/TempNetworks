from setuptools import setup, find_packages

setup(
    name="tempnets",
    version="0.1.0",
    description="Zigzag persistent homology of temporal networks",
    packages=find_packages(include=["tempnets", "tempnets.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "networkx",
        "dionysus",
    ],
    extras_require={
        # TemporalGraphComplex (classes.py) and tests/modelcomparison.py
        "extras": ["bats-tda", "persim"],
    },
)
