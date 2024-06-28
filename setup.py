"""Setup script for the DRAGON module."""

from setuptools import find_packages
from setuptools import setup

import badgers

setup(name='badgers',
      description='Package to use model-guided exploration of sequence space to diagnostic design guide RNAs for CRISPR-Cas13a',
      url="https://github.com/broadinstitute/badgers-cas13",
      version=1.0,
      packages=find_packages(),
      package_data={
        "badgers": ["utils/gan_data/*"],
      },
      scripts=[
        'design_guides.py'
      ]) 