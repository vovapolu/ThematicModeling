import numpy as np
import numpy.random
import time
import os.path
import os
import multiprocessing
import sys
from EM import *
from matplotlib import pyplot as plt

def write_plot_data(phi, phi0, theta, theta0, subjects, alpha, beta, id):
    pfilename = "plot.txt"

    rnum = len(subjects)
    subjects_dist = np.array([subject[2] for subject in subjects])
    min_dist = np.min(subjects_dist)
    max_dist = np.max(subjects_dist)
    avg_dist = np.average(subjects_dist)
    sparsity = float(np.count_nonzero(phi) + np.count_nonzero(theta)) / (phi.size + theta.size)
    t = len(theta)
    t0 = len(theta0)
    pfile = file(pfilename, "a")
    pfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n".format(
        id, rnum, min_dist, max_dist, avg_dist, sparsity, alpha, beta, t, t0
    ))
    pfile.close()


def write_special_plot_data(subjects, id):
    splot_file = file("splot.txt", "a")
    for subject in subjects:
        splot_file.write("{0}\t{1}\t{2}\t{3}\n".format(id, subject[0], subject[1], subject[2]))
    splot_file.close()

def check(tcount, dcount, wcount, alpha_factor, beta_factor, dirichlet_factor):
    phi, theta, ndw = generate_data(tcount, dcount, wcount, dirichlet_factor)

    alpha = np.full((wcount, ), alpha_factor)
    beta = np.full((tcount, ), beta_factor)
    start = time.clock()
    print "EM-Algorithm..."
    phi0, theta0 = EM_algorithm(ndw, wcount, tcount, alpha, beta)
    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    start = time.clock()
    print "Calculating Hellinger distance..."

    ntheta0 = normalized_matrix_by_rows(theta0)
    ntheta = normalized_matrix_by_rows(theta)
    res = compare_matrices(phi0, ntheta0, phi, ntheta)
    correct_phi0, correct_theta0, correct_product = constuct_correct_matrices(phi0, theta0, res)
    calculate_hellinger_dists(correct_phi0, correct_theta0, correct_product, phi, theta, phi.dot(theta))

    finish = time.clock()
    print "Calculated in ", "%.2f" % (finish - start), "seconds.", "\n"


def check_more_subjects(tcount, dcount, wcount, tcount0, is_generate_data, is_save_data,
                      alpha_factor, beta_factor, sparsity_factor, id):
    np.random.seed(id)
    if not os.path.exists("log"):
        os.makedirs("log")
    #sys.stdout = open("log/log_{0}.txt".format(id), "w")
    if is_generate_data:
        phi, theta, ndw = generate_data(tcount, dcount, wcount, sparsity_factor)
        if is_save_data:
            write_data_to_file("data", phi, theta, ndw)
    else:
        phi, theta, ndw = load_from_file("data")

    alpha = np.full((wcount, ), alpha_factor)
    beta = np.full((tcount, ), beta_factor)

    start = time.clock()
    print "EM-Algorithm..."

    phi0, theta0 = EM_algorithm(ndw, wcount, tcount0, alpha, beta)
    if is_save_data:
        np.savetxt("data_phi0.txt", phi0)
        np.savetxt("data_theta0.txt", theta0)

    finish = time.clock()
    print "Total time:", "%.2f" % (finish - start), "seconds.", "\n"

    theta = normalized_matrix_by_rows(theta)
    theta0 = normalized_matrix_by_rows(theta0)
    subjects = reconstruct_subjects(phi0, theta0, phi, theta)

    if not os.path.exists("dot"):
        os.makedirs("dot")
    dots_number = 0
    for subject in subjects:
        dots_number += 1
        np.savetxt("dot/dots" + str(dots_number) + ".txt", np.dstack((phi[:, subject[0]], phi0[:, subject[1]]))[0], delimiter='\t', fmt='%.10f')

    write_plot_data(phi, phi0, theta, theta0, subjects, alpha_factor, beta_factor, id)
    write_special_plot_data(subjects, id)


p = multiprocessing.Pool(processes=3)
processes = []
id = int(file("plot.txt").readlines()[-1].split()[0])
w = 1000
d = 1000
t = 250
t0 = 50
alpha_factor = 0.0
beta_factor = 0.0
dirichlet_factor = 0.01

def gen_alpha_data(mina, maxa, points_num, is_gen_matrices):
    global id
    for a in np.linspace(mina, maxa, num=points_num):
        id += 1
        processes.append(p.apply_async(check_more_subjects,
                                       args=(t, d, w, t0, is_gen_matrices, False, a, a, dirichlet_factor, id)))

    run_rocesses()


def gen_sparcity_data(mind, maxd, points_num, is_gen_matrices):
    global id
    for d in np.linspace(mind, maxd, num=points_num):
        id += 1
        processes.append(
            p.apply_async(check_more_subjects,
                          args=(t, d, w, t0, is_gen_matrices, False, alpha_factor, beta_factor, d, id)))

    run_rocesses()


def gen_tdata(mint, maxt, stept, mint0, maxt0, stept0, is_gen_matrices):
    global id
    for t0 in xrange(mint0, maxt0, stept0):
        for t in xrange(mint, maxt, stept):
            id += 1
            processes.append(
                p.apply_async(check_more_subjects,
                              args=(t, d, w, t0, is_gen_matrices, False, alpha_factor, beta_factor, dirichlet_factor, id)))

    run_rocesses()


def gen_special_data(points_num):
    global id
    id += 1
    check_more_subjects(t, w, d, t0, True, True, 0, 0, 0.01, id)
    for i in xrange(points_num):
        id += 1
        processes.append(p.apply_async(check_more_subjects,
                                       args=(t, w, d, t0, False, False, alpha_factor, beta_factor, dirichlet_factor, id)))

    run_rocesses()


def run_rocesses():
    global processes
    for p in processes:
        try:
            p.get()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print e

    processes = []

#check(t, d, w, alpha_factor, beta_factor, dirichlet_factor)
check_more_subjects(t, d, w, t0, True, False, 7000, 0, 0.01, id)
