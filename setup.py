import os

from setuptools import find_packages, setup

with open(os.path.join("hrl", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

""" 


setup(
    name="hrl",
    packages=find_packages(),
    package_data={"hrl": ["py.typed", "version.txt"]},
    install_requires=[
        "gym>=0.17",
        "numpy",
        "torch>=1.4.0",
    ],
    
    description=" Framework for hrl",
    author="Jiaming Guo, Yunkai Gao",
    url="https://github.com/guojm14/HRL",
    author_email="",
    license="MIT",
    version=__version__,
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
