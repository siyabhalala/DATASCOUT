from setuptools import setup, find_packages
setup(
    name="datascout",
    version="3.0.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
)
