import numpy as np

tsvFile = file("plot.txt", "r")
outFile = file("plotTmp.txt", "w")
columnsToExtract = [-2, -1, 1]
delimiter = "\t"
beginCharacter = ""
endCharacher = ""

lines = [map(float, line.split("\t")) for line in tsvFile]
extractedLines = sorted([[nums[i] for i in columnsToExtract] for nums in lines])

for nums in extractedLines:
    nums[-1] = nums[-1] / nums[-2]
    resStr = beginCharacter + delimiter.join(["{0}".format(num) for num in nums]) + endCharacher + "\n"
    outFile.write(resStr)



