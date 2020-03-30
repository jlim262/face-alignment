import io
import os
from os import path
import re
from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

VERSION = find_version('face_aligner', '__init__.py')

requirements = [
    'torch',
    'numpy',
    'scipy>=0.17',
    'scikit-image',
    'opencv-python',
    'tqdm',
    'enum34;python_version<"3.4"'
]

setup(
    name='face_aligner',
    version=VERSION,

    description="Face alignment tool from Python",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author details
    author="JongYoon Lim",
    author_email="jongyoon@apache.org",
    url="https://github.com/jlim262/face-alignment",

    # Package info
    packages=find_packages(exclude=('face_utils/test',)),

    install_requires=requirements,
    license='MIt',
    zip_safe=True,

    classifiers=[        
        'Operating System :: OS Independent',
        'License :: MIT License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 3.7',
    ],
)