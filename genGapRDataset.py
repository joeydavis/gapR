# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 05:08:20 2013

@author: Joey Davis : jhdavis@scripps.edu
"""

import numpy
import scipy
import matplotlib
import math
import random
import pylab


def makePoints(numPoints=4, center=[50,50], spread=50, r=0, ps=0):
    """makePoints is a helper function, it returns a list of points sampled uniformaly parameters described below.
        NOTE: makePoints updates a global variable COLOR, which is used to keep track of which points were generated together

    :param numPoints: The number of points to be generated
    :type numPoints: int
    :param center: The center of the distribution of points
    :type b: array (2D by default listed [x,y])
    :param spread: the size of the uniform distribution (each point defined as center+random (0-1)*spread)
    :type spread: float
    :param r: the current recursion depth when the points are made
    :type r: int
    :param ps: this parameter defines whether COLOR will be updated (the value should only be 1 if you are actually making points,
        if you are simply using the function to define new centers, ps should be set to 0)
    :type ps: int
    :returns:  a list of lists, each element of the form [x,y,color,recursionLevel]
    
    """
    global COLOR    
    COLOR = COLOR+ps
    return [[center[0]+(random.random()-1)*spread, center[1]+(random.random()-1)*spread, COLOR, r] for i in xrange(numPoints)]

def initCenter(center=[50,50], inSpread=50, r=0, numPoints=5, maxPoints=10, probContinue=0.5, fracContinueDrop=0.5):
    """initCenter is a recursive function that creates new centers for points to be generated at

    :param center: The [x,y] location of the center of the points to be distributed about
        Of the form [x,y]; defaults to [50,50]
    :type center: list
    :param inSpread: How wide points should be spread about the center
        Defaults to 50
    :type inSpread: float 
    :param r: a recursion counter to track how deep you've recursed
        Defaults to 0
    :type r: int
    :param numPoints: The average number of centers/points to generate (real number sampled from a normal dist with u=s=numPoints)
        Defaults to 5
    :type numPoints: int
    :param maxPoints: The maximum number of points to be generated
        Defaults to 10
    :type maxPoints: int
    :param probContinue: The probability that the points will be new centers
        (another round of recursion) instead of new points
        Defaults to 0.5
    :type probContinue: float
    :param fracContinueDrop: The factor by which the probContinue parameter will be dropped 
        if another round of recursion is called
        Defaults to 0.5
    :type fracContinueDrop: float
    :returns:  a list of lists, each element of the form [x,y,color,recursionLevel]
    
    """
    guess = float(random.random())
    if guess > float(probContinue):
        return makePoints(numPoints=min(getNorm(numPoints),maxPoints), center=center, spread=inSpread, r=r, ps=1)
        
    else:
        newCents = makePoints(numPoints=min(getNorm(numPoints),maxPoints), center=center, spread=inSpread, r=r, ps=0)
        allPoints = []
        for i in newCents:
            hold = initCenter(center=i, inSpread=inSpread/10, probContinue=probContinue*fracContinueDrop, r=r+1, numPoints=numPoints, maxPoints=maxPoints)
            for j in hold:
                allPoints.append(j)
        return allPoints

def getNorm(u, s=None):
    """getNorm returns an integer sample from a nomral distribution centered at u with a std. dev. of s

    :param u: center of the distribution
    :type u: int
    :param s: the std. dev. of the distribution
        Defaults to u if no std. dev. given
    :type u: int
    :returns:  an integer sampled from the normal distribution centered at u (std. dev. of u by default).
        If the chosen number is negative, 1 is returned.
        
    """
    if s is None:
        s = float(u)
    return max([int(math.floor(numpy.random.normal(u, s))),1])

def plotPoints(dataPoints, f, sizeScalar=75):
    """Generates a 2D scatter plot of the points given in dataPoints

    :param dataPoints: A list of lists. Each element is of the form:
        [xCoord, yCoord, color, size]
    :type dataPoints: list
    :param f: A figure to modify
    :type f: pylab.figure()
    :param sizeScalar: An int of the minimum dot size (also the scale for each level of recursion)
        Defaults to 75
    :type sizeScalar: int
    :returns:  The modified figure
        
    """
    xs = [x[0] for x in dataPoints]
    ys = [x[1] for x in dataPoints]
    cs = [x[2] for x in dataPoints]
    rs = [x[3] for x in dataPoints]

    rsf = [i*75+75 for i in rs]

    maxPoints = float(max(cs))
    csf = [float(i)/maxPoints for i in cs]
    
    ax = f.add_subplot(111)
    
    ax.scatter(xs, ys, c=csf, s=[max(rsf)-i+75 for i in rsf])

    for l_creationRound, l_recursionLevel, x, y in zip(cs, rs, xs, ys):
        ax.annotate(
            str(l_creationRound) + ":" + str(l_recursionLevel),
            xy = (x, y), xytext = (-5, 3),
            textcoords = 'offset points', ha = 'right', va = 'bottom')
    ax.set_xlabel('dim 1', fontsize=12)
    ax.set_ylabel('dim 2', fontsize=12)
    ax.set_title('Randomly generated points: labels = group:recursion depth')
    return f
"""#################Sample Execution Code###############
"""
global COLOR
figArray = [0]*5
for i in xrange(5):
    COLOR = 0
    dataPoints = initCenter(center=[100,100], inSpread=1000, r=0, numPoints=4, maxPoints=10, probContinue=1, fracContinueDrop=0.8)
    f = pylab.figure()
    f = plotPoints(dataPoints, f)
    figArray[i] = [f, dataPoints]

pylab.show('all')