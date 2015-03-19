import numpy as np

tsvFile = file("plot.txt", "r")
outFile = file("plotTmp.txt", "w")

columnsToSort = [-2, -1]
delimiter = "\t"
beginCharacter = ""
endCharacher = ""
minId = 501
maxId = 1000

def cmpLines(linea, lineb):
    aToSort = [linea[i] for i in columnsToSort]
    bToSort = [lineb[i] for i in columnsToSort]
    return cmp(aToSort, bToSort)

lines = [map(float, line.split("\t")) for line in tsvFile]
lines = [line for line in lines if line[0] >= minId and line[0] <= maxId]
sortedLines = sorted(lines, cmp=cmpLines)

for line in sortedLines:
    line[1] = line[1] / line[-1]
    id = int(line[-1])
    resStr = beginCharacter + delimiter.join(map(str, line)) + endCharacher + "\n"
    outFile.write(resStr)



