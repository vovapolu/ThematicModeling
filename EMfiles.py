import numpy as np


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


def load_from_file(filename, alg):
    alg.phi = np.loadtxt(filename + "_phi.txt")
    alg.theta = np.loadtxt(filename + "_theta.txt")
    with file(filename + "_ndw.txt", "r") as infile:
        size = int(infile.readline())
        alg.ndw = []
        for i in xrange(size):
            slice_shape = tuple(map(int, infile.readline().split()))
            lines = [infile.readline() for i in xrange(slice_shape[0])]
            new_slice = np.array(map(lambda s: map(int, s.split()), lines))
            alg.ndw.append(new_slice)


def write_data_to_file(filename, alg):
    np.savetxt(filename + "_phi.txt", alg.phi)
    np.savetxt(filename + "_theta.txt", alg.theta)
    with file(filename + "_ndw.txt", "w") as outfile:
        outfile.write('{0}\n'.format(len(alg.ndw)))
        for ndwSlice in alg.ndw:
            outfile.write('{0} {1}\n'.format(len(ndwSlice), len(ndwSlice[0])))
            np.savetxt(outfile, ndwSlice, fmt='%i')


def write_plot_data(alg, alpha, beta, id):
    pfilename = "plot.txt"

    rnum = len(alg.reconstructed_themes)
    themes_dist = np.array([theme[2] for theme in alg.reconstructed_themes])
    min_dist = np.min(themes_dist)
    max_dist = np.max(themes_dist)
    avg_dist = np.average(themes_dist)
    sparsity = float(np.count_nonzero(alg.phi) + np.count_nonzero(alg.theta)) / (alg.phi.size + alg.theta.size)
    pfile = file(pfilename, "a")
    pfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n".format(
        id, rnum, min_dist, max_dist, avg_dist, sparsity, alpha, beta, alg.tcount, alg.t0count
    ))
    pfile.close()


def write_special_plot_data(themes, id):
    splot_file = file("splot.txt", "a")
    for theme in themes:
        splot_file.write("{0}\t{1}\t{2}\t{3}\n".format(id, theme[0], theme[1], theme[2]))
    splot_file.close()