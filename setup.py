# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@package    tcmi

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import io
import json
import tcmi as pkg
from setuptools import setup, find_packages

with io.open('metainfo.json', encoding='utf-8') as file:
    metainfo = json.load(file)

setup(
    name=pkg.__name__,
    version=pkg.__version__,
    author=', '.join(metainfo['authors']),
    author_email=metainfo['email'],
    url=metainfo['url'],
    description=metainfo['title'],
    long_description=metainfo['description'],
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib'],
)