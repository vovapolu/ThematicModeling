import numpy as np
import numpy.random
import time
import os.path
import os
import multiprocessing
import sys
from EM import *

def writePlotData(phi, phi0, theta, theta0, subjects, alpha, beta, id):
    pfilename = "plot.txt"

    rnum = len(subjects)
    subjectsDist = [subject[2] for subject in subjects]
    minDist = min(subjectsDist)
    maxDist = max(subjectsDist)
    avgDist = np.average(subjectsDist)
    sparsity = float(np.count_nonzero(phi) + np.count_nonzero(theta)) / (phi.size + theta.size)
    t = len(theta)
    t0 = len(theta0)
    pfile = file(pfilename, "a")
    pfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n".format(
        id, rnum, minDist, maxDist, avgDist, sparsity, alpha, beta, t, t0
    ))
    pfile.close()


def writeSpecialPlotData(subjects, id):
    splotFile = file("splot.txt", "a")
    for subject in subjects:
        splotFile.write("{0}\t{1}\t{2}\t{3}\n".format(id, subject[0], subject[1], subject[2]))
    splotFile.close()


def check(tCount, dCount, wCount):
    phi, theta, ndw = generateData(tCount, dCount, wCount, 0.5)

    alpha = [0] * wCount
    beta = [0] * tCount
    start = time.clock()
    print "EM-Algorithm..."
    phi0, theta0 = EM(ndw, wCount, tCount, alpha, beta)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Calculating Hellinger distance..."

    nTheta0 = normTheta(theta0)
    nTheta = normTheta(theta)
    res = compareMatrices(phi0, nTheta0, phi, nTheta)
    correctPhi0, correctTheta0, correctProduct = constuctCorrectMatrices(phi0, theta0, res)
    #subjects = reconstructSubjects(phi0, theta0, phi, theta, res)
    calculateHellingerDists(correctPhi0, correctTheta0, correctProduct, phi, theta, phi.dot(theta))

    finish = time.clock()
    print "Calculated in ", "%.2f" % (finish - start), "seconds.", "\n"


def checkMoreSubjects(tCount, dCount, wCount, tCountEM, isGeneratingData, isSaveData,
                      alphaFactor, betaFactor, sparsityFactor, id):
    np.random.seed(id)
    if not os.path.exists("log"):
        os.makedirs("log")
    sys.stdout = open("log/log_{0}.txt".format(id), "w")
    if isGeneratingData:
        phi, theta, ndw = generateData(tCount, dCount, wCount, sparsityFactor)
        if (isSaveData):
            writeDataToFile("data", phi, theta, ndw)
    else:
        phi, theta, ndw = loadFromFile("data")

    alpha = [alphaFactor] * wCount
    beta = [betaFactor] * tCount
    #alpha[-1] = 0.00
    #beta[-1] = 0.00
    start = time.clock()
    print "EM-Algorithm..."
    phi0, theta0 = EM(ndw, wCount, tCountEM, alpha, beta)
    if (isSaveData):
        np.savetxt("data_phi0.txt", phi0)
        np.savetxt("data_theta0.txt", theta0)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    
    theta = normTheta(theta)
    theta0 = normTheta(theta0)
    subjects = reconstructSubjects(phi0, theta0, phi, theta)
    writePlotData(phi, phi0, theta, theta0, subjects, alphaFactor, betaFactor, id)
    writeSpecialPlotData(subjects, id)


p = multiprocessing.Pool(processes=3)
processes = []
id = int(file("plot.txt").readlines()[-1].split()[0])
w = 300
d = 300
t = 50
t0 = 50

def genAlphaData(minA, maxA, pointsNum, isGenMatrices):
    for a in np.linspace(minA, maxA, num=pointsNum):
        id += 1
        processes.append(p.apply_async(checkMoreSubjects, args=(w, d, t, t0, isGenMatrices, False, a, a, 0.01, id)))

    runProcesses()


def genSparcityData(minD, maxD, pointsNum, isGenMatrices):
    for d in np.linspace(minD, maxD, num=pointsNum):
        id += 1
        processes.append(
            p.apply_async(checkMoreSubjects, args=(w, d, t, t0, isGenMatrices, False, -0.02, -0.02, d, id)))

    runProcesses()


def genTData(minT, maxT, stepT, minT0, maxT0, stepT0, isGenMatrices):
    for t0 in xrange(minT0, maxT0, stepT0):
        for t in xrange(minT, maxT, stepT):
            id += 1
            processes.append(
                p.apply_async(checkMoreSubjects, args=(t, w, d, t0, isGenMatrices, False, -0.02, -0.02, 0.01, id)))

    runProcesses()


def genSpecialData(pointsNum):
    global id
    id += 1
    checkMoreSubjects(t, w, d, t0, True, True, -0.02, -0.02, 0.01, id)
    for i in xrange(pointsNum):
        id += 1
        processes.append(p.apply_async(checkMoreSubjects, args=(t, w, d, t0, False, False, -0.02, -0.02, 0.01, id)))

    runProcesses()


def runProcesses():
    global processes
    for p in processes:
        try:
            p.get()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print e

    processes = []

checkMoreSubjects(t, w, d, t0, True, False, 0, 0, 0.5, id)
