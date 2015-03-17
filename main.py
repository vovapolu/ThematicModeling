import numpy as np
import numpy.random
import numpy.linalg
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from scipy.spatial.distance import euclidean
from scipy import stats
import matplotlib.pyplot as plot
import numpy.random.mtrand as mr
import os.path
import os
import multiprocessing
import sys
from collections import deque

def normalized(a):
    v = sum(a)
    return a / v

def pround(a):
    frac = a - np.floor(a)
    if (np.random.random_sample() < 1 - frac):
        return np.floor(a)
    else:
        return np.ceil(a)

proundVec = np.vectorize(pround)

def positiveSlice(a):
    if (a < 0):
        return 0
    else:
        return a

positiveSliceVec = np.vectorize(positiveSlice)

def cutZeroes(m):
    zeroEps = 0.00001
    m[np.abs(m) < zeroEps] = 0

def getVersion():
    try:
        with file("version.txt", "r") as vFile:
            ver = int(vFile.readline())
    except IOError:
        ver = 0
    return ver

def incrementVersion():
    ver = getVersion()
    with file("version.txt", "w") as vFile:
        vFile.write("{0}".format(ver + 1))
    return ver + 1

def loadFromFile(filename):
    phi = np.loadtxt(filename + "_phi.txt")
    theta = np.loadtxt(filename + "_theta.txt")
    with file(filename + "_ndw.txt", "r") as infile:
        size = int(infile.readline())
        ndw = []
        for i in xrange(size):
            sliceShape = tuple(map(int, infile.readline().split()))
            lines = [infile.readline() for i in xrange(sliceShape[0])]
            newSlice = np.array(map(lambda s: map(int, s.split()), lines))
            ndw.append(newSlice)
    return (phi, theta, ndw)

def writeDataToFile(filename, phi, theta, ndw):
    np.savetxt(filename + "_phi.txt", phi)
    np.savetxt(filename + "_theta.txt", theta)
    with file(filename + "_ndw.txt", "w") as outfile:
        outfile.write('{0}\n'.format(len(ndw)))
        for ndwSlice in ndw:
            outfile.write('{0} {1}\n'.format(len(ndwSlice), len(ndwSlice[0])))
            np.savetxt(outfile, ndwSlice, fmt='%i')

def genPhi(wCount, tCount, dirichletFactor):
    phi = np.empty((wCount, tCount))
    phi[:, -1] = np.fromfunction(lambda i: 1.0 / (i + 1), (wCount, ), dtype=float) * (0.8 + np.random.random((wCount, )) * 0.4)
    phi[:,:-1] = mr.dirichlet((-phi[:, -1] + phi[0, -1]) * dirichletFactor, tCount - 1).transpose()
    cutZeroes(phi)
    phi = np.apply_along_axis(normalized, 0, phi)
    return phi

def genTheta(tCount, dCount, dirichletFactor):
    theta = np.empty((tCount, dCount))
    theta[-1] = np.random.random((dCount, )) * 3
    theta[:-1, :] = mr.dirichlet([dirichletFactor] * (tCount - 1), dCount).transpose()
    cutZeroes(theta)
    theta = np.apply_along_axis(normalized, 0, theta)
    return theta

def genCollection(phi, theta):
    tCount = len(theta)
    dCount = len(theta[0])
    nd = mr.randint(600, 1000, dCount)
    pwd = phi.dot(theta)
    ndw = []
    for d in xrange(dCount):
        columnD = proundVec(pwd[:, d] * nd[d]).astype(int)
        columnD = np.dstack((np.arange(len(columnD)), columnD))[0]
        columnD1 = columnD[:, 1]
        ndw.append(columnD[np.abs(columnD1) > 0, :])
    return ndw

def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

def hellinger2Matrix(a, b):
    return np.average([hellinger2(a[:, i], b[:, i]) for i in xrange(len(a[0]))])

def norm(phi, theta):
    return np.linalg.norm(phi) / np.product(phi.shape) + np.linalg.norm(theta) / np.product(theta.shape)

def compareMatrices(phi0, theta0, phi, theta):
    costMatrix = np.empty((len(theta0), len(theta)))
    for i in xrange(len(theta0)):
        for j in xrange(len(theta)):
            costMatrix[i][j] = hellinger2(theta0[i], theta[j]) + hellinger2(phi0[:, i], phi[:, j])
    print "Hungarian algorithm..."
    start = time.clock()
    res = linear_assignment(costMatrix)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"
    return res

def EM(ndw, wCount, tCount, alpha, beta):
    dCount = len(ndw)
    print "W =", wCount
    print "T =", tCount
    print "D =", dCount
    theta = np.random.random((tCount, dCount))
    phi = np.random.random((wCount, tCount))
    for i in xrange(dCount):
        theta[:, i] = normalized(theta[:, i])
    for i in xrange(tCount):
        phi[:, i] = normalized(phi[:, i])
    nwt = np.empty((wCount, tCount))
    ndt = np.empty((dCount, tCount))
    nd = np.empty(dCount)
    nt = np.empty(tCount)
    for d in xrange(dCount):
        nd[d] = ndw[d][:, 1].sum()

    eps = 0.000015

    isConverged = False
    step = 0

    while not isConverged:
        start = time.clock()
        thetaOld = theta.copy()
        phiOld = phi.copy()
        nwt.fill(0)
        ndt.fill(0)

        for d in xrange(dCount):
            for wInd in xrange(len(ndw[d])):
                w, wVal = (ndw[d][wInd][0], ndw[d][wInd][1])
                delta = phi[w] * theta[: ,d]
                delta = delta * wVal / sum(delta)
                nwt[w] += delta
                ndt[d] += delta

        for w in xrange(wCount):
            phi[w] = positiveSliceVec(nwt[w] + alpha[w])
        phi = np.apply_along_axis(normalized, 0, phi)

        for t in xrange(tCount):
            theta[t] = positiveSliceVec(ndt[:, t] + beta[t])
        theta = np.apply_along_axis(normalized, 0, theta)

        step += 1
        nowNorm = norm(phi - phiOld, theta - thetaOld)
        finish = time.clock()
        print "[EM-Algorithm] Step:", step, "Norm:", nowNorm
        print "[EM-Algorithm]", "%.2f" % (finish - start), "seconds."
        isConverged = nowNorm < eps

    return phi, theta

def generateData(tCount, dCount, wCount, sparsityFactor):
    start = time.clock()
    print "Generating theta..."
    theta = genTheta(tCount, dCount, sparsityFactor)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Generating phi..."
    phi = genPhi(wCount, tCount, sparsityFactor)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Generating collection..."
    ndw = genCollection(phi, theta)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"
    return (phi, theta, ndw)

def constuctCorrectMatrices(phi0, theta0, res):
    correctPhi0 = np.empty(phi0.shape)
    correctTheta0 = np.empty(theta0.shape)
    for e in res:
        correctPhi0[:, e[1]] = phi0[:, e[0]]
    for e in res:
        correctTheta0[e[1]] = theta0[e[0]]
    correctProduct = correctPhi0.dot(correctTheta0)
    return (correctPhi0, correctTheta0, correctProduct)

def calculateHellingerDists(correctPhi0, correctTheta0, correctProduct, phi, theta, product):
    distPhi = hellinger2Matrix(correctPhi0, phi)
    distTheta = hellinger2Matrix(correctTheta0, theta)
    distProduct = hellinger2Matrix(correctProduct, product)
    print "Hellinger distance for phi:", distPhi
    print "Hellinger distance for theta:", distTheta
    print "Hellinger distance for product:", distProduct
    return (distPhi, distTheta, distProduct)

def reconstructSubjects(phi0, theta0, phi, theta, res):
    expectedValTheta = np.sqrt(len(theta0[0])) / (np.sqrt(2) * 3)
    expectedValPhi = np.sqrt(len(phi0[:, 0])) / (np.sqrt(2) * 3)
    diffsPhi = np.array([hellinger2(phi0[:, e[0]], phi[:, e[1]]) for e in res]) / expectedValPhi
    diffsTheta = np.array([hellinger2(theta0[e[0]], theta[e[1]]) for e in res]) / expectedValTheta
    allDiffs = diffsPhi + diffsTheta
    print len(allDiffs[allDiffs < 1]), "subjects were reconstructed."
    print np.arange(len(allDiffs))[allDiffs < 1]
    print allDiffs

    return (np.arange(len(allDiffs))[allDiffs < 1], allDiffs[allDiffs < 1])

def check(tCount, dCount, wCount):
    phi, theta, ndw = generateData(tCount, dCount, wCount)

    alpha = [0.01] * wCount
    beta = [0.01] * tCount
    start = time.clock()
    print "EM-Algorithm..."
    phi0, theta0 = EM(ndw, wCount, tCount, alpha, beta)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Calculating Hellinger distance..."

    res = compareMatrices(phi0, theta0, phi, theta)
    correctPhi0, correctTheta0, correctProduct = constuctCorrectMatrices(phi0, theta0, res)
    subjects = reconstructSubjects(phi0, theta0, phi, theta, res)
    calculateHellingerDists(correctPhi0, correctTheta0, correctProduct, phi, theta, phi.dot(theta))

    finish = time.clock()
    print "Calculated in ", "%.2f" % (finish - start), "seconds.", "\n"

def writePlotData(phi, phi0, theta, theta0, subjects, alpha, beta, id):
    pfilename = "plot.txt"
    if (not os.path.exists(pfilename)):
        with file(pfilename, "w") as pfile:
            pfile.write("#Id RNum MinDist MaxDist AvgDist Sparsity RegA RegB T T0\n")

    rnum = len(subjects[0])
    minDist = min(subjects[1])
    maxDist = max(subjects[1])
    avgDist = np.average(subjects[1])
    sparsity = float(np.count_nonzero(phi) + np.count_nonzero(theta)) / (phi.size + theta.size)
    t = len(theta)
    t0 = len(theta0)
    pfile = file(pfilename, "a")
    pfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n".format(
        id, rnum, minDist, maxDist, avgDist, sparsity, alpha, beta, t, t0
    ))
    pfile.close()


def checkMoreSubjects(tCount, dCount, wCount, tCountEM, isGeneratingData, isSaveData,
                      alphaFactor, betaFactor, sparsityFactor, id):
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
    alpha[-1] = 0.00
    beta[-1] = 0.00
    start = time.clock()
    print "EM-Algorithm..."
    phi0, theta0 = EM(ndw, wCount, tCountEM, alpha, beta)
    if (isSaveData):
        np.savetxt("data_phi0.txt", phi0)
        np.savetxt("data_theta0.txt", theta0)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    res = compareMatrices(phi0, theta0, phi, theta)
    subjects = reconstructSubjects(phi0, theta0, phi, theta, res)
    with file("results.txt", "a") as afile:
        afile.write("Data version: {0}\n".format(id))
        afile.write("T = {0}, T0 = {1}, D = {2}, W = {3}\n".format(tCount, tCountEM, dCount, wCount))
        afile.write("{0} subjects were reconstructed.\n".format(len(subjects[0])))
        np.savetxt(afile, subjects[0], fmt="%i", newline=" ")
        afile.write("\n")
        np.savetxt(afile, subjects[1], fmt="%.2f", newline=" ")
        afile.write("\n")
    writePlotData(phi, phi0, theta, theta0, subjects, alphaFactor, betaFactor, id)


p = multiprocessing.Pool(processes=3)
processes = []
id = 200
#for a in np.linspace(-0.2, 0.2, num=40):
#    id += 1
#    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, True, False, a, a, 0.01, id)))

#for d in np.linspace(0.001, 1.5, num=20):
#    id += 1
#    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, True, False, 0, 0, d, id)))

#id += 1
#checkMoreSubjects(100, 1000, 1000, 10, True, True, 0.01, 0.01, 0.01, id)

#for a in np.linspace(-0.3, 0.3, num=20):
#    id += 1
#    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, False, False, a, a, 0.05, id)))

#for d in np.linspace(0.001, 0.2, num=20):
#    id += 1
#    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, True, False, 0, 0, d, id)))

for t0 in xrange(5, 50, 4):
    for t in xrange(100, 1001, 100):
    	id += 1
    	processes.append(p.apply_async(checkMoreSubjects, args=(t, 1000, 1000, t0, True, False, -0.02, -0.02, 0.01, id)))

for p in processes:
    try:
        p.get()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e

processes = []

id += 1
checkMoreSubjects(100, 1000, 1000, 10, True, True, 0.01, 0.01, 0.01, id)

for a in np.linspace(-0.4, 0.4, num=30):
    id += 1
    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, False, False, a, a, 0.05, id)))

for d in np.linspace(0.001, 0.5, num=30):
    id += 1
    processes.append(p.apply_async(checkMoreSubjects, args=(100, 1000, 1000, 10, True, False, 0, 0, d, id)))

for p in processes:
    try:
        p.get()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        
