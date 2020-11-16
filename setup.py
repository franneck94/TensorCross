from setuptools import setup


def get_readme() -> str:
    with open("README.md") as f:
        return f.read()


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
AUTHOR = "Jan Schaffranek, Saif Al-Dilaimi"
DESCRIPTION = "Cross Validation, Grid Search and Random Search for TensorFlow Datasets."
LICENSE = get_license()
README = get_readme()

VERSION = '0.1.0'
ISRELEASED = False

PYTHON_MIN_VERSION = "3.7"
PYTHON_MAX_VERSION = "3.8"

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    long_description=README,
    packages=["tensorcross", "tests"],
    author=AUTHOR,
    description=DESCRIPTION,
    classifiers=[CLASSIFIERS],
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
