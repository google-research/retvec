"""
 Copyright 2021 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

setup(name="retvec",
      version=get_version('retvec/__init__.py'),
      description="Resilient and Efficient Text Vectorizer",
      long_description=long_description,
      author='Elie Bursztein',
      author_email='github@elie.net',
      url='https://github.com/google-research/retvect',
      license='Apache License 2.0',
      install_requires=[
          "tensorflow>=2.6", "tabulate", "numpy", "tqdm",
          "tensorflow_similarity", "google-cloud",
          "matplotlib", "tabulate"
      ],
      extras_require={
          "benchmarks": [
              "huggingface", "datasets", "tokenizers"
          ],
          "dev": ["mypy", "pytest", "flake8",
                  "pytest-cov", "twine"]
      },
      classifiers=[
          'Development Status :: 3 - Alpha', 'Environment :: Console',
          'Framework :: TensorFlow',
          'License :: OSI Approved :: Apache Software License',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      packages=find_packages())
