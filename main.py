#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:05:30 2018

@author: kyle
"""
import operator

from filter_node import *
from deap import gp, creator, base, tools, algorithms

pset = gp.PrimitiveSet("MAIN", 1)

pset.addPrimitive(mean, 1)
pset.addPrimitive(equalizeHist, 1)
pset.addPrimitive(normalization, 1)
pset.addPrimitive(erode, 1)
pset.addPrimitive(dilate, 1)
pset.addPrimitive(sobel, 1)
pset.addPrimitive(lightEdge, 1)
pset.addPrimitive(darkEdge, 1)
pset.addPrimitive(lightPixel, 1)
pset.addPrimitive(darkPixel, 1)
pset.addPrimitive(largeArea, 1)
pset.addPrimitive(smallArea, 1)
pset.addPrimitive(inversion, 1)
pset.addPrimitive(logicalProd, 2)
pset.addPrimitive(logicalSum, 2)
pset.addPrimitive(algebraicProd, 2)
pset.addPrimitive(algebraicSum, 2)
pset.addPrimitive(boundedProd, 2)
pset.addPrimitive(boundedSum, 2)
pset.renameArguments(ARGO='image')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def imageLoader():
    trainingImageFolder = "TrainingDataSet/TrainingImages/"
    targetImageFolder = "TrainingDataSet/TargetImages/"
    trainingImageNames = ("3.bmp", "34.bmp", "35.bmp", "8.bmp")
    targetImageNames = ("3.jpg", "34.jpg", "35.jpg", "8.jpg")

    imageSet = []
    for trainingName, targetName in zip(trainingImageNames, targetImageNames):
        trainingImage = cv2.imread(trainingImageFolder + trainingName, cv2.IMREAD_GRAYSCALE)
        targetImage = 255 - imread(targetImageFolder + targetName, cv2.IMREAD_GRAYSCALE)
        imageSet.append([trainingImage, targetImage])
    return imageSet

def evalFilter(individual, image):
    global toolbox
    func = toolbox.compile(expr=individual)
    absResidual = cv2.absdiff(func(image[0]), image[1])
    absResidual = np.float(absResidual)
    weightedImage = cv2.compare(absResidual, 64, cv2.CMP_LE) + cv2.multiply(cv2.compare(absResidual, 64, cv2.CMP_GT), 0.5)
    weightedImage = np.float(weightedImage)
    return cv2.sumElems(cv2.multiply(weightedImage, absResidual))/cv2.sumElems(weightedImage*255)

toolbox.register("evaluate", evalFilter, images)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)