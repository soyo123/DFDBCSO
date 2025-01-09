# Import packages
import os
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy

# Global parameters
PopSize = 200
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
pbest = None
pbest_fit = None
curFEs = 0
FuncNum = 1
curIter = 0
MaxIter = int(MaxFEs / PopSize * 2)
phi = 0.1

# Initialize population randomly
def Initialization(func):
    global Pop, Velocity, FitPop, pbest, pbest_fit
    Velocity = np.zeros((PopSize, DimSize))
    pbest = np.zeros((PopSize, DimSize))
    pbest_fit = np.full(PopSize, np.inf)  # Initial fitness values as infinity
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])
        pbest[i] = Pop[i]
        pbest_fit[i] = FitPop[i]

# Boundary check for individuals
def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi

# Calculate FDB scores dynamically
def calculate_fdb_scores(Pop, pbest, FitPop, alpha):
    """
    Calculate dynamic FDB scores based on distance to pbest and fitness
    """
    distances = np.linalg.norm(Pop - pbest, axis=1)
    normalized_distances = distances / np.max(distances)  # Normalize distances
    normalized_fitness = FitPop / np.max(FitPop)          # Normalize fitness
    fdb_scores = (1 - alpha) * normalized_distances + alpha * normalized_fitness
    return fdb_scores

# Select high and low scoring pairs
def select_high_low_pairs(fdb_scores):
    sorted_indices = np.argsort(fdb_scores)
    low_indices = sorted_indices[:len(sorted_indices)//2]
    high_indices = sorted_indices[len(sorted_indices)//2:]
    np.random.shuffle(low_indices)
    np.random.shuffle(high_indices)
    pairs = list(zip(high_indices, low_indices))
    return pairs

# Update pbest
def update_pbest():
    global Pop, FitPop, pbest, pbest_fit
    for i in range(PopSize):
        if FitPop[i] < pbest_fit[i]:  # Assume a minimization problem
            pbest[i] = Pop[i]
            pbest_fit[i] = FitPop[i]

# Dynamic FDB CSO algorithm
def DFDBCSO(func):
    global Pop, Velocity, FitPop, phi, curIter, MaxIter, pbest, pbest_fit

    alpha = curIter / MaxIter
    fdb_scores = calculate_fdb_scores(Pop, pbest, FitPop, alpha)
    pairs = select_high_low_pairs(fdb_scores)

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xmean = np.mean(Pop, axis=0)

    for idx1, idx2 in pairs:
        if FitPop[idx1] < FitPop[idx2]:
            Off[idx1] = deepcopy(Pop[idx1])
            FitOff[idx1] = FitPop[idx1]
            Velocity[idx2] = np.random.rand(DimSize) * Velocity[idx2] + np.random.rand(DimSize) * (
                Pop[idx1] - Pop[idx2]) + phi * (Xmean - Pop[idx2])
            Off[idx2] = Pop[idx2] + Velocity[idx2]
            Off[idx2] = Check(Off[idx2])
            FitOff[idx2] = func(Off[idx2])
        else:
            Off[idx2] = deepcopy(Pop[idx2])
            FitOff[idx2] = FitPop[idx2]
            Velocity[idx1] = np.random.rand(DimSize) * Velocity[idx1] + np.random.rand(DimSize) * (
                Pop[idx2] - Pop[idx1]) + phi * (Xmean - Pop[idx1])
            Off[idx1] = Pop[idx1] + Velocity[idx1]
            Off[idx1] = Check(Off[idx1])
            FitOff[idx1] = func(Off[idx1])

    Pop = deepcopy(Off)
    FitPop = deepcopy(FitOff)

    # Update pbest
    update_pbest()

# Run DFDBCSO algorithm
def RunDFDBCSO(func):
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        BestList = []
        curIter = 0
        np.random.seed(1998 + 18 * i)
        Initialization(func)
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            DFDBCSO(func)
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt("./DFDBCSO/DFDBCSO_Data/CEC2022/F" + str(FuncNum+1) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")

# Main function to execute DFDBCSO
def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]

    for i in range(len(CEC2022)):
        FuncNum = i
        RunDFDBCSO(CEC2022[i].evaluate)

# Entry point
if __name__ == "__main__":
    if not os.path.exists('./DFDBCSO/DFDBCSO_Data/CEC2022'):
        os.makedirs('./DFDBCSO/DFDBCSO_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
