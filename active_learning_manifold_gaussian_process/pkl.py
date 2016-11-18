# -*- coding: utf-8 -*-

import cPickle


def _save(fname, data):
    with open(fname, "wb") as f:
        cPickle.dump(data, f)

def _load(fname):
    with open(fname, "rb") as f:
        return cPickle.load(f)

