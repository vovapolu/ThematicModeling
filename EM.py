import numpy as np
import numpy.random
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from scipy.spatial.distance import euclidean
import numpy.random.mtrand as mr
import os.path


def normalized(v):
    s = float(np.sum(v))
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


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)


def hellinger2_matrix(a, b):
    return np.average([hellinger2(a[:, i], b[:, i]) for i in xrange(len(a[0]))])


def theme_dist(phi0, theta0, phi, theta, i, j):
    return hellinger2(np.append(theta0[i], phi0[:, i]), np.append(theta[j], phi[:, j])) ** 2 / 2


class Algorithm:
    def __init__(self):
        self.timers = {}

    def out(self, *strs):
        print "[" + self.__class__.__name__ + "]", " ".join([str(p) for p in strs])

    def start_timer(self, timer_name):
        self.timers[timer_name] = time.clock()
        self.last_timer = timer_name

    def finish_timer(self, timer_name=None):
        if timer_name is None:
            timer_name = self.last_timer
        self.out(timer_name, "{0:0.2f} s".format(time.clock() - self.timers[timer_name]))


class EMAlgorithm(Algorithm):

    def __init__(self, wcount, dcount, tcount, t0count, algrorithm_steps=50):
        Algorithm.__init__(self)
        self.wcount = wcount
        self.dcount = dcount
        self.tcount = tcount
        self.t0count = t0count
        self.algorithm_steps = algrorithm_steps

    def gen_theta(self, dirichlet_factor):
        self.theta = np.empty((self.tcount, self.dcount))
        self.theta[-1] = np.random.random((self.dcount, )) * 3
        self.theta[:-1, :] = mr.dirichlet([dirichlet_factor] * (self.tcount - 1), self.dcount).transpose()
        cut_zeroes(self.theta)
        self.theta = normalized_matrix_by_columns(self.theta)

    def gen_phi(self, dirichlet_factor):
        self.phi = np.empty((self.wcount, self.tcount))
        self.phi[:, -1] = np.fromfunction(lambda i: 1.0 / (i + 1), (self.wcount, ), dtype=float) * (
            0.8 + np.random.random((self.wcount, )) * 0.4)
        self.phi[:, :-1] = mr.dirichlet(1 / self.phi[:, -1] / self.wcount * dirichlet_factor, self.tcount - 1).transpose()
        cut_zeroes(self.phi)
        self.phi = normalized_matrix_by_columns(self.phi)

    def gen_collection(self):
        nd = mr.randint(600, 1000, self.dcount)

        proportion = np.sum(normalized(nd) * self.theta[-1] * np.sum(self.phi[:, -1]))
        self.out("Background theme word proportion:", proportion)

        pwd = self.phi.dot(self.theta)
        self.ndw = []
        for d in xrange(self.dcount):
            columnd = pround(pwd[:, d] * nd[d])
            columnd_tup = np.dstack((np.arange(len(columnd)), columnd))[0]
            self.ndw.append(columnd_tup[columnd > 0, :])

    def generate_data(self, dirichlet_factor):
        self.start_timer("Generating theta:")
        self.gen_theta(dirichlet_factor)
        self.finish_timer()

        self.start_timer("Generating phi:")
        self.gen_phi(dirichlet_factor)
        self.finish_timer()

        self.start_timer("Generating collection:")
        self.gen_collection()
        self.finish_timer()

    def process(self, alpha, beta):
        self.start_timer("EM-Algorithm:")
        self.out("W =", self.wcount)
        self.out("T =", self.t0count)
        self.out("D =", self.dcount)

        self.theta0 = np.random.random((self.t0count, self.dcount))
        self.phi0 = np.random.random((self.wcount, self.t0count))
        self.theta0 = normalized_matrix_by_columns(self.theta0)
        self.phi0 = normalized_matrix_by_columns(self.phi0)

        nwt = np.empty((self.wcount, self.t0count))
        ndt = np.empty((self.dcount, self.t0count))
        nd = np.empty(self.dcount)

        for d in xrange(self.dcount):
            nd[d] = self.ndw[d][:, 1].sum()

        step = 0

        for i in xrange(self.algorithm_steps):
            self.start_timer("Iteration {}:".format(i))
            nwt.fill(0)
            ndt.fill(0)

            for d in xrange(self.dcount):
                for wInd in xrange(len(self.ndw[d])):
                    w, w_val = self.ndw[d][wInd][0], self.ndw[d][wInd][1]
                    delta = self.phi0[w] * self.theta0[:, d]
                    delta_sum = np.sum(delta)
                    if delta_sum > 0:
                        delta = delta * w_val / delta_sum
                        nwt[w] += delta
                        ndt[d] += delta

            #for w in xrange(self.wcount):
            #    phi[w] = positive_slice(nwt[w] + alpha[w])
            for w in xrange(self.wcount):
                self.phi0[w] = positive_slice(nwt[w] - alpha[0] * self.phi0[w] * (np.sum(self.phi0[w]) - self.phi0[w]))

            for t in xrange(self.t0count):
                self.theta0[t] = positive_slice(ndt[:, t] + beta[t])

            self.theta0 = normalized_matrix_by_columns(self.theta0)
            self.phi0 = normalized_matrix_by_columns(self.phi0)

            step += 1
            self.finish_timer()
        self.finish_timer("EM-Algorithm:")

    def normalize_thetas(self):
        self.normalize_theta = normalized_matrix_by_rows(self.theta)
        self.normalize_theta0 = normalized_matrix_by_rows(self.theta0)

    def compare_matrices(self):
        cost_matrix = np.empty((self.t0count, self.tcount))
        for i in xrange(self.t0count):
            for j in xrange(self.tcount):
                cost_matrix[i][j] = theme_dist(self.phi0, self.normalize_theta0, self.phi, self.normalize_theta, i, j)
        self.start_timer("Hungarian algorithm:")
        self.theme_assigment = linear_assignment(cost_matrix)
        self.finish_timer()

    def constuct_correct_matrices(self):
        self.correct_phi0 = np.empty(self.phi0.shape)
        self.correct_theta0 = np.empty(self.theta0.shape)
        for asgn in self.theme_assigment:
            self.correct_phi0[:, asgn[1]] = self.phi0[:, asgn[0]]
        for asgn in self.theme_assigment:
            self.correct_theta0[asgn[1]] = self.theta0[asgn[0]]

    def calculate_hellinger_dists(self):
        dist_phi = hellinger2_matrix(self.correct_phi0, self.phi)
        dist_theta = hellinger2_matrix(self.correct_theta0, self.theta)
        dist_product = hellinger2_matrix(self.correct_phi0.dot(self.correct_theta0), self.phi.dot(self.theta))
        self.out("Hellinger distance for phi:", dist_phi)
        self.out("Hellinger distance for theta:", dist_theta)
        self.out("Hellinger distance for product:", dist_product)

    def reconstruct_themes(self):
        self.reconstructed_themes = []
        for i in xrange(self.t0count):
            dists_t = [theme_dist(self.phi0, self.theta0, self.phi, self.theta, i, j) for j in xrange(self.tcount)]
            min_ind = np.argmin(dists_t)
            dists_t0 = [theme_dist(self.phi0, self.theta0, self.phi, self.theta, j, min_ind) for j in xrange(self.t0count)]
            min_ind0 = np.argmin(dists_t0)
            if min_ind0 == i:
                self.reconstructed_themes.append((min_ind, i, dists_t[min_ind]))