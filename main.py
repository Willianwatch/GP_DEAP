#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:05:30 2018

@author: kyle
"""

import filt
from deap import gp

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(filt.mean, 1)
pset.addPrimitive(filt.equalizeHist, 1)
pset.addPrimitive(filt.normalization, 1)
pset.addPrimitive(filt.erode, 1)
pset.addPrimitive(filt.dilate, 1)
pset.addPrimitive(filt.sobel, 1)
pset.addPrimitive(filt.lightEdge, 1)
pset.addPrimitive(filt.darkEdge, 1)
pset.addPrimitive(filt.lightPixel, 1)
pset.addPrimitive(filt.darkPixel, 1)
pset.addPrimitive(filt.largeArea, 1)
pset.addPrimitive(filt.smallArea, 1)
pset.addPrimitive(filt.inversion, 1)
pset.addPrimitive(filt.logicalProd, 2)
pset.addPrimitive(filt.logicalSum, 2)
pset.addPrimitive(filt.algebraicProd, 2)
pset.addPrimitive(filt.algebraicSum, 2)
pset.addPrimitive(filt.boundedProd, 2)
pset.addPrimitive(filt.boundedSum, 2)