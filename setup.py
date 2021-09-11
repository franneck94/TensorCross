from tensorcross.version import __version__

from setuptools import find_packages
from setuptools import setup


CLASSIFIERS = """\
License :: OSI Approved
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Software Development
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

DISTNAME = "tensorcross"
AUTHORS = "Jan Schaffranek, Saif Al-Dilaimi"
DESCRIPTION = (
    "Cross Validation, Grid Search and Random Search "
    "for TensorFlow Datasets."
)
LICENSE = "MIT"
README = (
    "Cross Validation, Grid Search and Random Search for TensorFlow "
    "Datasets. For more information see here: "
    "https://github.com/franneck94/TensorCross"
)

VERSION = __version__
ISRELEASED = False

MIN_PYTHON_VERSION = "3.7"
MAX_PYTHON_VERSION = "3.9"
INSTALL_REQUIRES = ["tensorflow>=2.0", "numpy", "scipy", "scikit-learn"]

PACKAGES = find_packages(include=["tensorcross", "tensorcross.*"])

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    author=AUTHORS,
    python_requires=f">={MIN_PYTHON_VERSION},<={MAX_PYTHON_VERSION}",
    install_requires=INSTALL_REQUIRES,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
