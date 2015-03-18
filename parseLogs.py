import os
import numpy as np
import re

outfile = file("logs.txt", "w")

for log in os.listdir("log"):
    num = int(re.search("(?<=_)\d+", log).group(0))
    print num
    logFile = file("log/" + log, "r")
    try:
        lines = logFile.readlines()
        i = -1
        while (lines[i].find("[") == -1):
            i -= 1
        vals = lines[i]
        nums = map(float, vals.strip("[ ]\n").split())
        outfile.write("{0}\t{1}\t{2}\t{3}\n".format(num, min(nums), min(nums), np.average(nums)))
    except IndexError as e:
        print e
    except ValueError as e:
        print e
