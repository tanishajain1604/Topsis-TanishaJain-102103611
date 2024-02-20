#importing necessary libraries
from math import inf
import numbers
import sys
import numpy as np
import pandas as pd
#step1.1
def vectorNormalization(inputFileName):
    #calculate root of sum of squares to normalise and scale down the values to a range
    root_sum_square = [0] * (inputFileName.shape[1])

    for i in range(1, inputFileName.shape[1]):
        sum_ = 0
        for j in range(0, len(inputFileName.iloc[:, i])):
            sum_ += pow(inputFileName.iloc[j, i], 2)

        root_sum_square[i] = np.sqrt(sum_)

    norm = inputFileName.copy()
#step1.2
    #normalising all values in table
    for i in range(1, norm.shape[1]):
        for j in range(0, len(norm.iloc[:, i])):
            # Divide every  value by its Root of sum of squares
            norm.iloc[j, i] /= root_sum_square[i]

    result = "\nNormalised Decision Metrix:\n" + str(norm)
    return norm, result
#step1.3
def WeightAssignment(norm, Weights):
    # calculate weight * Normalized performance value
    for i in range(1, norm.shape[1]):
        for j in range(0, len(norm.iloc[:, i])):
            norm.iloc[j, i] *= Weights[i-1]

    result = "\nWeighted Normalized Decision Matrix:\n" + str(norm)
    return norm, result

#step2
#best and worst values
def calculateVjs(norm, Impacts):
    # calculate ideal best value and ideal worst value
    # -ve means: min is best
    # +ve means: max is best
    vjpositive = [0] * (norm.shape[1]-1)
    vjnegative = [0] * (norm.shape[1]-1)

    for i in range(1, norm.shape[1]):
        if Impacts[i-1] == '+':
            vjpositive[i-1] = -inf
            vjnegative[i-1] = inf

            for j in range(0, len(norm.iloc[:, i])):
                vjpositive[i-1] = max(vjpositive[i-1], norm.iloc[j, i])
                vjnegative[i-1] = min(vjnegative[i-1], norm.iloc[j, i])

        else:
            vjpositive[i-1] = inf
            vjnegative[i-1] = -inf

            for j in range(0, len(norm.iloc[:, i])):
                vjpositive[i-1] = min(vjpositive[i-1], norm.iloc[j, i])
                vjnegative[i-1] = max(vjnegative[i-1], norm.iloc[j, i])

    result = "\nvjpositive: " + str(vjpositive) + "\nvjnegative: " + str(vjnegative)
    return vjpositive, vjnegative, result

#step3
def calculateSIs(norm, vjpositive, vjnegative):
    # Calculate Euclidean distance from ideal best value and ideal worst value
    sipos = [0] * (norm.shape[0])
    sineg = [0] * (norm.shape[0])

    for i in range(0, norm.shape[0]):
        sum_ = 0
        sum2 = 0
        for j in range(1, norm.shape[1]):
            sum_ += pow(norm.iloc[i, j] - vjpositive[j-1], 2)
            sum2 += pow(norm.iloc[i, j] - vjnegative[j-1], 2)

        sipos[i] = np.sqrt(sum_)
        sineg[i] = np.sqrt(sum2)

    result = "\nsipos: " + str(sipos) + "\nsineg: " + str(sineg)
    return sipos, sineg, result

#step4 main topsis function to consolidate everything 
def topsis(inputFileName, Weights, Impacts, resultFileName):
    norm, norm_result = vectorNormalization(inputFileName)
    norm, weight_result = WeightAssignment(norm, Weights)
    vjpositive, vjnegative, vj_result = calculateVjs(norm, Impacts)
    sipos, sineg, si_result = calculateSIs(norm, vjpositive, vjnegative)

    # Calculate Performance score
    for i in range(0, inputFileName.shape[0]):
        inputFileName.loc[i, "Topsis Score"] = sineg[i] / (sipos[i] + sineg[i])

    inputFileName["Rank"] = inputFileName["Topsis Score"].rank(method='max')

    result = "\n\n Dataset with Topsis Scores and rank\n" + str(inputFileName)
    print(f"Saving to {resultFileName}")
    inputFileName.to_csv(resultFileName, index=False)#final result

def is_numeric(n):
    for i in range(0, len(n)-1):
        if not isinstance(n[i], numbers.Real):
            return False
    return True

def checkInputs(inputFileName, Weights, Impacts, resultFileName):
    if not inputFileName.endswith('.csv'):
        sys.exit("Only csv files permitted\n")

    try:
        data = pd.read_csv(inputFileName)
    except FileNotFoundError:
        print("File not Found")
    else:
        col = data.shape[1]
        if col < 3:
            sys.exit("input file must contain 3 or more columns\n")

        for i in range(1, col):
            if not is_numeric(data.iloc[:, i]):
                sys.exit("Data type should be numeric")

        if ',' not in Weights:
            sys.exit("Weights must be separated by comma\n")

        Weights = Weights.split(",")
        if len(Weights) != col-1:
            sys.exit(f"Specify {col-1} weights\n")

        if ',' not in Impacts:
            sys.exit("Impacts must be separated by comma\n")

        Impacts = Impacts.split(",")
        if len(Impacts) != col-1:
            sys.exit(f"Specify {col-1} impacts\n")

        if set(Impacts) != {'+', '-'}:
            sys.exit("Only '+', '-' impacts are allowed\n")

        if not resultFileName.endswith('.csv'):
            sys.exit("Only csv files permitted\n")

        return data, Weights, Impacts

def main():
    n = len(sys.argv)
    if n != 5:
        print("Incorrect number of arguments\n")
        return

    inputFileName = sys.argv[1]
    Weights = sys.argv[2]
    Impacts = sys.argv[3]
    resultFileName = sys.argv[4]

    inputFileName, Weights, Impacts = checkInputs(inputFileName, Weights, Impacts, resultFileName)
    Weights = [eval(i) for i in Weights]

    topsis(inputFileName, Weights, Impacts, resultFileName)

if __name__ == "__main__":
    main()
