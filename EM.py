import numpy as np
import numpy.random
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from scipy.spatial.distance import euclidean
import numpy.random.mtrand as mr
import os.path

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
    phi[:, -1] = np.fromfunction(lambda i: 1.0 / (i + 1), (wCount, ), dtype=float) * (
        0.8 + np.random.random((wCount, )) * 0.4)
    phi[:, :-1] = mr.dirichlet((-phi[:, -1] + phi[0, -1]) * dirichletFactor, tCount - 1).transpose()
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

def normTheta(theta):
    nTheta = np.empty(theta.shape)
    for t in xrange(theta.shape[0]):
        for d in xrange(theta.shape[1]):
            nTheta[t][d] = theta[t][d] / sum(theta[t, :])
    return nTheta
        
def themeDist(phi0, theta0, phi, theta, i, j):
    return hellinger2(np.append(theta0[i], phi0[:, i]), np.append(theta[j], phi[:, j])) ** 2 / 2

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

    eps = 0.05

    isConverged = False
    step = 0

    for i in xrange(50):
        start = time.clock()
        thetaOld = theta.copy()
        phiOld = phi.copy()
        nwt.fill(0)
        ndt.fill(0)

        for d in xrange(dCount):
            for wInd in xrange(len(ndw[d])):
                w, wVal = (ndw[d][wInd][0], ndw[d][wInd][1])
                delta = phi[w] * theta[:, d]
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
        nowNorm = hellinger2Matrix(phi, phiOld) + hellinger2Matrix(theta, thetaOld)
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

def calculateHellingerDists(correctPhi0, correctTheta0, correctProduct, phi, theta, product):
    distPhi = hellinger2Matrix(correctPhi0, phi)
    distTheta = hellinger2Matrix(correctTheta0, theta)
    distProduct = hellinger2Matrix(correctProduct, product)
    print "Hellinger distance for phi:", distPhi
    print "Hellinger distance for theta:", distTheta
    print "Hellinger distance for product:", distProduct
    return (distPhi, distTheta, distProduct)
    
def constuctCorrectMatrices(phi0, theta0, res):
    correctPhi0 = np.empty(phi0.shape)
    correctTheta0 = np.empty(theta0.shape)
    for e in res:
        correctPhi0[:, e[1]] = phi0[:, e[0]]
    for e in res:
        correctTheta0[e[1]] = theta0[e[0]]
    correctProduct = correctPhi0.dot(correctTheta0)
    return (correctPhi0, correctTheta0, correctProduct)

def compareMatrices(phi0, theta0, phi, theta):
    costMatrix = np.empty((len(theta0), len(theta)))
    for i in xrange(len(theta0)):
        for j in xrange(len(theta)):
            costMatrix[i][j] = themeDist(phi0, theta0, phi, theta, i, j)
    print "Hungarian algorithm..."
    start = time.clock()
    res = linear_assignment(costMatrix)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"
    return res

def reconstructSubjects(phi0, theta0, phi, theta):
    t = phi.shape[1]
    t0 = phi0.shape[1]
    res = []   
    for i in xrange(t0):
        distsT = [themeDist(phi0, theta0, phi, theta, i, j) for j in xrange(t)]
        minInd = np.argmin(distsT)
        distsT0 = [themeDist(phi0, theta0, phi, theta, j, minInd) for j in xrange(t0)]
        minInd0 = np.argmin(distsT0)
        print i, minInd, distsT[minInd], minInd0, distsT0[minInd0]
        if (minInd0 == i):
            res.append((minInd, i, distsT[minInd]))
    
    return res