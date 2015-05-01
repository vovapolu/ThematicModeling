import numpy as np
import numpy.random
import time
import os.path
import os
import multiprocessing
import sys
from EM import *
from matplotlib import pyplot as plt
import EMfiles


def check(alg, alpha_factor, beta_factor, dirichlet_factor):
    alg.generate_data(dirichlet_factor)

    alpha = np.full((alg.wcount, ), alpha_factor)
    beta = np.full((alg.tcount, ), beta_factor)
    alg.process(alpha, beta)

    alg.normalize_thetas()
    alg.compare_matrices()
    alg.constuct_correct_matrices()
    alg.calculate_hellinger_dists()

def check_more_subjects(alg, is_generate_data, is_save_data,
                      alpha_factor, beta_factor, dirichlet_factor, id):
    np.random.seed(id)
    if not os.path.exists("log"):
        os.makedirs("log")
    #sys.stdout = open("log/log_{0}.txt".format(id), "w")
    if is_generate_data:
        alg.generate_data(dirichlet_factor)
        if is_save_data:
            EMfiles.write_data_to_file("data", alg)
    else:
        EMfiles.load_from_file("data", alg)

    alpha = np.full((alg.wcount, ), alpha_factor)
    beta = np.full((alg.tcount, ), beta_factor)
    alg.process(alpha, beta)

    if is_save_data:
        np.savetxt("data_phi0.txt", alg.phi0)
        np.savetxt("data_theta0.txt", alg.theta0)

    alg.normalize_thetas()
    alg.reconstruct_themes()

    if not os.path.exists("dot"):
        os.makedirs("dot")
    dots_number = 0
    for theme in alg.reconstructed_themes:
        dots_number += 1
        np.savetxt("dot/dots" + str(dots_number) + ".txt", np.dstack((alg.phi[:, theme[0]], alg.phi0[:, theme[1]]))[0], delimiter='\t', fmt='%.10f')

    EMfiles.write_plot_data(alg, alpha_factor, beta_factor, id)
    EMfiles.write_special_plot_data(alg.reconstructed_themes, id)


p = multiprocessing.Pool(processes=3)
processes = []
id = int(file("plot.txt").readlines()[-1].split()[0])


def gen_alpha_data(mina, maxa, points_num, is_gen_matrices):
    global id
    for a in np.linspace(mina, maxa, num=points_num):
        id += 1
        processes.append(p.apply_async(check_more_subjects,
                                       args=(alg, is_gen_matrices, False, a, a, dirichlet_factor, id)))

    run_rocesses()


def gen_sparcity_data(mind, maxd, points_num, is_gen_matrices):
    global id
    for d in np.linspace(mind, maxd, num=points_num):
        id += 1
        processes.append(
            p.apply_async(check_more_subjects,
                          args=(alg, is_gen_matrices, False, alpha_factor, beta_factor, d, id)))

    run_rocesses()


def gen_tdata(mint, maxt, stept, mint0, maxt0, stept0, is_gen_matrices):
    global id
    for t0 in xrange(mint0, maxt0 + 1, stept0):
        for t in xrange(mint, maxt + 1, stept):
            id += 1
            processes.append(
                p.apply_async(check_more_subjects,
                              args=(alg, is_gen_matrices, False, alpha_factor, beta_factor, dirichlet_factor, id)))

    run_rocesses()


def gen_special_data(points_num):
    global id
    id += 1
    check_more_subjects(alg, True, True, alpha_factor, beta_factor, dirichlet_factor, id)
    for i in xrange(points_num):
        id += 1
        processes.append(p.apply_async(check_more_subjects,
                                       args=(alg, False, False, alpha_factor, beta_factor, dirichlet_factor, id)))

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

alg = EMAlgorithm(wcount=500, dcount=500, tcount=250, t0count=50)
alpha_factor = 7000.0
beta_factor = 0.0
dirichlet_factor = 0.01

#check(t, d, w, alpha_factor, beta_factor, dirichlet_factor)
check_more_subjects(alg, True, False, 4000, 0, 0.01, id)
#gen_tdata(250, 250, 1, 50, 250, 20, True)