from tensorcross.version import __version__

from setuptools import find_packages
from setuptools import setup


CLASSIFIERS = """\
License :: OSI Approved
Programming Language :: Python :: 3.8
Topic :: Software Development
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

DISTNAME = "tensorcross"
AUTHORS = "Jan Schaffranek, Saif Al-Dilaimi"
DESCRIPTION = ("Cross Validation, Grid Search and Random Search "
               "for TensorFlow Datasets.")
LICENSE = "MIT"
README = ("Cross Validation, Grid Search and Random Search for TensorFlow "
          "Datasets. For more information see here: "
          "https://github.com/franneck94/TensorCross")

VERSION = __version__
ISRELEASED = False

PYTHON_VERSION = "3.8"
INSTALL_REQUIRES = [
    "tensorflow==2.3.1",
    "numpy==1.18.*",
    "scikit-learn"
]

PACKAGES = find_packages(include=["tensorcross", "tensorcross.*"])

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    author=AUTHORS,
    python_requires=f"=={PYTHON_VERSION}",
    install_requires=INSTALL_REQUIRES,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
