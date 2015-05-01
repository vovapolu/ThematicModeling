import numpy as np
import numpy.random
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from scipy.spatial.distance import euclidean
import numpy.random.mtrand as mr
import os.path


def normalized(v):
    s = np.sum(v)
    if s != 0:
        return v / s


def normalized_matrix_by_columns(m):
    return np.apply_along_axis(normalized, 0, m)


def normalized_matrix_by_rows(m):
    return np.apply_along_axis(normalized, 1, m)


def pround(v):
    isRound = np.random.random(size=v.shape) > v - np.floor(v)
    v[isRound] += 1
    return np.floor(v).astype(int)


def positive_slice(a):
    a[a < 0] = 0
    return a


def cut_zeroes(m):
    zero_eps = 0.00001
    m[np.abs(m) < zero_eps] = 0


def get_version():
    try:
        with file("version.txt", "r") as vFile:
            ver = int(vFile.readline())
    except IOError:
        ver = 0
    return ver


def increment_version():
    ver = get_version()
    with file("version.txt", "w") as vFile:
        vFile.write("{0}".format(ver + 1))
    return ver + 1


def load_from_file(filename):
    phi = np.loadtxt(filename + "_phi.txt")
    theta = np.loadtxt(filename + "_theta.txt")
    with file(filename + "_ndw.txt", "r") as infile:
        size = int(infile.readline())
        ndw = []
        for i in xrange(size):
            slice_shape = tuple(map(int, infile.readline().split()))
            lines = [infile.readline() for i in xrange(slice_shape[0])]
            new_slice = np.array(map(lambda s: map(int, s.split()), lines))
            ndw.append(new_slice)
    return phi, theta, ndw


def write_data_to_file(filename, phi, theta, ndw):
    np.savetxt(filename + "_phi.txt", phi)
    np.savetxt(filename + "_theta.txt", theta)
    with file(filename + "_ndw.txt", "w") as outfile:
        outfile.write('{0}\n'.format(len(ndw)))
        for ndwSlice in ndw:
            outfile.write('{0} {1}\n'.format(len(ndwSlice), len(ndwSlice[0])))
            np.savetxt(outfile, ndwSlice, fmt='%i')


def gen_phi(wcount, tcount, dirichlet_factor):
    phi = np.empty((wcount, tcount))
    phi[:, -1] = np.fromfunction(lambda i: 1.0 / (i + 1), (wcount, ), dtype=float) * (
        0.8 + np.random.random((wcount, )) * 0.4)
    phi[:, :-1] = mr.dirichlet(1 / phi[:, -1] / wcount * dirichlet_factor, tcount - 1).transpose()
    cut_zeroes(phi)
    phi = np.apply_along_axis(normalized, 0, phi)
    return phi


def gen_theta(tcount, dcount, dirichlet_factor):
    theta = np.empty((tcount, dcount))
    theta[-1] = np.random.random((dcount, )) * 3
    theta[:-1, :] = mr.dirichlet([dirichlet_factor] * (tcount - 1), dcount).transpose()
    cut_zeroes(theta)
    theta = np.apply_along_axis(normalized, 0, theta)
    return theta


def gen_collection(phi, theta):
    tcount = len(theta)
    dcount = len(theta[0])
    nd = mr.randint(600, 1000, dcount)
    pwd = phi.dot(theta)
    ndw = []
    for d in xrange(dcount):
        columnd = pround(pwd[:, d] * nd[d])
        columnd_tup = np.dstack((np.arange(len(columnd)), columnd))[0]
        ndw.append(columnd_tup[columnd > 0, :])
    return ndw


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)


def hellinger2_matrix(a, b):
    return np.average([hellinger2(a[:, i], b[:, i]) for i in xrange(len(a[0]))])


def theme_dist(phi0, theta0, phi, theta, i, j):
    return hellinger2(np.append(theta0[i], phi0[:, i]), np.append(theta[j], phi[:, j])) ** 2 / 2


def EM_algorithm(ndw, wcount, tcount, alpha, beta):
    dcount = len(ndw)
    print "W =", wcount
    print "T =", tcount
    print "D =", dcount
    theta = np.random.random((tcount, dcount))
    phi = np.random.random((wcount, tcount))
    theta = normalized_matrix_by_columns(theta)
    phi = normalized_matrix_by_columns(phi)
    nwt = np.empty((wcount, tcount))
    ndt = np.empty((dcount, tcount))
    nd = np.empty(dcount)
    nt = np.empty(tcount)
    for d in xrange(dcount):
        nd[d] = ndw[d][:, 1].sum()

    step = 0

    for i in xrange(50):
        start = time.clock()
        nwt.fill(0)
        ndt.fill(0)


        for d in xrange(dcount):
            for wInd in xrange(len(ndw[d])):
                w, w_val = ndw[d][wInd][0], ndw[d][wInd][1]
                delta = phi[w] * theta[:, d]
                delta_sum = np.sum(delta)
                if delta_sum > 0:
                    delta = delta * w_val / delta_sum
                    nwt[w] += delta
                    ndt[d] += delta

        #for w in xrange(wcount):
        #    phi[w] = positive_slice(nwt[w] + alpha[w])
        for w in xrange(wcount):
            phi[w] = positive_slice(nwt[w] - alpha[0] * phi[w] * (np.sum(phi[w]) - phi[w]))

        for t in xrange(tcount):
            theta[t] = positive_slice(ndt[:, t] + beta[t])

        theta = normalized_matrix_by_columns(theta)
        phi = normalized_matrix_by_columns(phi)

        step += 1

        finish = time.clock()
        print "[EM-Algorithm] Step:", step
        print "[EM-Algorithm]", "%.2f" % (finish - start), "seconds."

    return phi, theta


def generate_data(tcount, dcount, wcount, sparsity_factor):
    start = time.clock()
    print "Generating theta..."
    theta = gen_theta(tcount, dcount, sparsity_factor)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Generating phi..."
    phi = gen_phi(wcount, tcount, sparsity_factor)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Generating collection..."
    ndw = gen_collection(phi, theta)
    finish = time.clock()
    print "Generated in", "%.2f" % (finish - start), "seconds.", "\n"
    return phi, theta, ndw


def calculate_hellinger_dists(correct_phi0, correct_theta0, correct_product, phi, theta, product):
    dist_phi = hellinger2_matrix(correct_phi0, phi)
    dist_theta = hellinger2_matrix(correct_theta0, theta)
    dist_product = hellinger2_matrix(correct_product, product)
    print "Hellinger distance for phi:", dist_phi
    print "Hellinger distance for theta:", dist_theta
    print "Hellinger distance for product:", dist_product
    return dist_phi, dist_theta, dist_product


def constuct_correct_matrices(phi0, theta0, res):
    correct_phi0 = np.empty(phi0.shape)
    correct_theta0 = np.empty(theta0.shape)
    for e in res:
        correct_phi0[:, e[1]] = phi0[:, e[0]]
    for e in res:
        correct_theta0[e[1]] = theta0[e[0]]
    correct_product = correct_phi0.dot(correct_theta0)
    return correct_phi0, correct_theta0, correct_product


def compare_matrices(phi0, theta0, phi, theta):
    cost_matrix = np.empty((theta0.shape[0], theta.shape[0]))
    for i in xrange(theta0.shape[0]):
        for j in xrange(theta.shape[0]):
            cost_matrix[i][j] = theme_dist(phi0, theta0, phi, theta, i, j)
    print "Hungarian algorithm..."
    start = time.clock()
    res = linear_assignment(cost_matrix)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"
    return res


def reconstruct_subjects(phi0, theta0, phi, theta):
    t = phi.shape[1]
    t0 = phi0.shape[1]
    res = []   
    for i in xrange(t0):
        dists_t = [theme_dist(phi0, theta0, phi, theta, i, j) for j in xrange(t)]
        min_ind = np.argmin(dists_t)
        dists_t0 = [theme_dist(phi0, theta0, phi, theta, j, min_ind) for j in xrange(t0)]
        min_ind0 = np.argmin(dists_t0)
        if min_ind0 == i:
            res.append((min_ind, i, dists_t[min_ind]))
    
    return res