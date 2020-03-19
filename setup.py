import setuptools
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

setuptools.setup(
    name="ABBA",
    version="0.0.1",
    author="Steven Elsworth <steven.elsworth@manchester.ac.uk>, Stefan Guettel <stefan.guettel@manchester.ac.uk>",
    description="A symbolic time series representation building Brownian bridges",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nla-group/ABBA",
    packages=setuptools.find_packages(),
    install_requires=['joblib','numpy','scikit-learn','scipy','coverage', 'matplotlib']
)
