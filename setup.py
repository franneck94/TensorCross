from setuptools import setup


def get_license() -> str:
    with open("LICENSE") as f:
        return f.read()


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
LICENSE = get_license()
README = ("Cross Validation, Grid Search and Random Search for TensorFlow "
          "Datasets. For more information see here: "
          "https://github.com/franneck94/TensorCross")

VERSION = '0.1.1'
ISRELEASED = False

PYTHON_VERSION = "3.8"

PACKAGES = [
    "tensorcross",
    "tests"
]

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=PACKAGES,
    author=AUTHORS,
    python_requires=f"=={PYTHON_VERSION}",
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
