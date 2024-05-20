from setuptools import find_packages
from setuptools import setup

from tensorcross.version import __version__


CLASSIFIERS = """\
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
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

MIN_PYTHON_VERSION = "3.9"
INSTALL_REQUIRES = [
    "tensorflow>=2.13",
    "keras>=3.0",
    "numpy",
    "scipy",
    "scikit-learn>=1.0",
]


PACKAGES = find_packages(include=["tensorcross", "tensorcross.*"])

metadata = dict(  # noqa: C408
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    author=AUTHORS,
    python_requires=f">={MIN_PYTHON_VERSION}",
    install_requires=INSTALL_REQUIRES,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
