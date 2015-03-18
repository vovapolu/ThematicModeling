import numpy as np

resFile = file("results.txt", "r")
lines = resFile.readlines()

for i in xrange(0, len(lines), 5):
    dataVersion = int(lines[i].split()[2])
    initVals = lines[i + 1].split()
    t = int(initVals[2].replace(",", ""))
    t0 = int(initVals[5].replace(",", ""))
    d = int(initVals[8].replace(",", ""))
    w = int(initVals[11].replace(",", ""))
    rSubjects = int(lines[i + 2].split()[0])
    nums = map(int, lines[i + 3].split())
    vals = map(float, lines[i + 4].split())
    print np.average(vals)


