# -*- coding: utf-8 -*-
"""
@package    tcmi.cache

@copyright  Copyright (c) 2018+ Fritz Haber Institute of the Max Planck Society,
            Benjamin Regler <regler@fhi-berlin.mpg.de>
@license    See LICENSE file for details.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
"""

import os
import gzip
import pickle
import tempfile


class Constant(tuple):
    "Pretty display of immutable constant."
    def __new__(cls, name):
        return tuple.__new__(cls, (name,))

    def __repr__(self):
        return '%s' % self[0]


class Cache(object):
    """Simple read-only disk-cache. 
    """

    ENOVAL = Constant('ENOVAL')

    def __init__(self, directory=None):
        """Initialize cache instance.
        """

        if directory is None:
            directory = tempfile.mkdtemp(prefix='diskcache-')
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)

        self._directory = directory
        self._filename = os.path.join(directory, 'cache.dat')
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory, 0o755)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise RuntimeError(
                        error.errno,
                        'Cache directory "%s" does not exist'
                        ' and could not be created' % self._directory
                    )

        data = {}
        if os.path.exists(self._filename):
            with gzip.open(self._filename, 'rb') as file:
                data = pickle.load(file)
        self._data = data
                
    @property
    def directory(self):
        """Cache directory."""
        return self._directory

    @property
    def filename(self):
        """Cache filename."""
        return self._filename

    def __getitem__(self, key):
        """Return corresponding value for `key` from cache.
        """
        value = self.get(key, default=self.ENOVAL)
        if value is self.ENOVAL:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        """Set corresponding `value` for `key` in cache.
        """
        self.set(key, value)

    def get(self, key, default=None):
        """Retrieve value from cache. If `key` is missing, return `default`.
        """
        return self._data.get(key, default)

    def set(self, key, value):
        """Set `key` and `value` item in cache.
        """
        self._data[key] = value
        return True

    def save(self):
        """Save cache to file.
        """
        with gzip.open(self._filename, 'wb') as file:
            pickle.dump(self._data, file, pickle.HIGHEST_PROTOCOL)
