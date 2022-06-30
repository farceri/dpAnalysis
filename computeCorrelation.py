'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import shapeDescriptors
import sys
import os

def pbcDistance(r1, r2, boxSize):
    delta = r1 - r2
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

def computeDistances(pos, boxSize):
    distances = np.zeros((pos.shape[0], pos.shape[0]))
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            delta = pbcDistance(pos[i], pos[j], boxSize)
            distances[i,j] = np.linalg.norm(delta)
    return distances

def computeParticleVelocities(vel, nv):
    numParticles = nv.shape[0]
    pVel = np.zeros((numParticles,2))
    firstVertex = 0
    for pId in range(numParticles):
        pVel[pId] = [np.mean(vel[firstVertex:firstVertex+nv[pId],0]), np.mean(vel[firstVertex:firstVertex+nv[pId],1])]
        firstVertex += nv[pId]
    return pVel

def computeCorrFunctions(pos1, pos2, boxSize, waveVector, scale, oneDim = False):
    #delta = np.linalg.norm(pbcDistance(pos1, pos2, boxSize), axis=1)
    delta = np.linalg.norm(pos1 - pos2, axis=1)
    if(oneDim == True):
        delta = pos1[:,0] - pos2[:,0]
    delta -= np.mean(delta)
    msd = np.mean(delta**2)
    isf = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    chi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isf*isf
    return msd / scale, isf, chi4

def computeSusceptibility(pos1, pos2, field, scale):
    delta = pos1[:,0] - pos2[:,0]
    delta -= np.mean(delta)
    sus = delta / field
    chi = np.mean(sus)
    return chi / scale

def computeShapeCorrFunction(shape1, shape2):
    return (np.mean(shape1 * shape2) - np.mean(shape1)**2) / (np.mean(shape1**2) - np.mean(shape1)**2)

def computeVelCorrFunction(vel1, vel2):
    return np.mean(np.sum(vel1 * vel2, axis=1))

def getDirectories(dirName):
    listDir = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "bab" and dir != "dynamics")):
            listDir.append(dir)
    return listDir

def readFromParams(dirName, paramName):
    with open(dirName + os.sep + "params.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                return float(scalarString)

def checkPair(dirName, index1, index2):
    if(os.path.exists(dirName + os.sep + "t" + str(index1))):
        if(os.path.exists(dirName + os.sep + "t" + str(index2))):
            return True
    return False

def readParticlePair(dirName, index1, index2):
    pPos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particlePos.dat"))
    pPos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particlePos.dat"))
    return pPos1, pPos2

def readPair(dirName, index1, index2):
    pPos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particlePos.dat"))
    pos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "positions.dat"))
    pPos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particlePos.dat"))
    pos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "positions.dat"))
    return pPos1, pos1, pPos2, pos2

def readVelPair(dirName, index1, index2):
    pVel1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleVel.dat"))
    pVel2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleVel.dat"))
    return pVel1, pVel2

def readDirectorPair(dirName, index1, index2):
    pAngle1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleAngles.dat"))
    pAngle2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleAngles.dat"))
    pVel1 = np.array([np.cos(pAngle1), np.sin(pAngle1)]).T
    pVel2 = np.array([np.cos(pAngle2), np.sin(pAngle2)]).T
    return pVel1, pVel2

def readShapePair(dirName, index1, index2):
    perimeter1 = np.loadtxt(dirName + os.sep + "t" + str(index1) + "/perimeters.dat")
    area1 = np.loadtxt(dirName + os.sep + "t" + str(index1) + "/areas.dat")
    perimeter2 = np.loadtxt(dirName + os.sep + "t" + str(index2) + "/perimeters.dat")
    area2 = np.loadtxt(dirName + os.sep + "t" + str(index2) + "/areas.dat")
    return perimeter1**2/(4*np.pi*area1), perimeter2**2/(4*np.pi*area2)

def calcShapePair(dirName, index1, index2, boxSize, nv):
    area1 = np.loadtxt(dirName + os.sep + "t" + str(index1) + "/areas.dat")
    pos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "positions.dat"))
    _, perimeter1 = shapeDescriptors.getAreaAndPerimeterList(pos1, boxSize, nv)
    np.savetxt(dirName + os.sep + "t" + str(index1) + os.sep + "perimeters.dat", perimeter1)
    area2 = np.loadtxt(dirName + os.sep + "t" + str(index2) + "/areas.dat")
    pos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "positions.dat"))
    _, perimeter2 = shapeDescriptors.getAreaAndPerimeterList(pos2, boxSize, nv)
    np.savetxt(dirName + os.sep + "t" + str(index2) + os.sep + "perimeters.dat", perimeter2)
    return perimeter1**2/(4*np.pi*area1), perimeter2**2/(4*np.pi*area2)

def readShape(dirName, boxSize, nv):
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    area = np.loadtxt(dirName + os.sep + "areas.dat")
    _, perimeter = shapeDescriptors.getAreaAndPerimeterList(pos, boxSize, nv)
    np.savetxt(dirName + os.sep + "perimeters.dat", perimeter)
    return perimeter**2/(4*np.pi*area)

def plotErrorBar(ax, x, y, err, xlabel, ylabel, logx = False, logy = False):
    ax.errorbar(x, y, err, marker='o', color='k', markersize=7, markeredgecolor='k', markeredgewidth=0.7, linewidth=1.2, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')
    plt.tight_layout()

def plotCorrelation(x, y, ylabel, xlabel = "$Distance,$ $r$", logy = False, logx = False, color = 'k', show = True):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.plot(x, y, linewidth=1.5, color=color, marker='.')
    if(logy == True):
        ax.set_yscale('log')
    if(logx == True):
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    plt.tight_layout()
    if(show == True):
        #plt.pause(0.5)
        plt.show()

########################### Pair Correlation Function ##########################
def computePairCorr(dirName, plot=True):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = readFromParams(dirName, "phi")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    meanRad = np.mean(rad)
    pos = np.loadtxt(dirName + os.sep + "particlePos.dat")
    pos = np.array(pos)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    distance = computeDistances(pos[rad>np.mean(rad)], boxSize).flatten()
    distance = distance[distance>0]
    bins = np.linspace(np.min(distance), np.max(distance), 50)
    pairCorr, edges = np.histogram(distance, bins=bins, density=True)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    pairCorr /= (phi * 2 * np.pi * binCenter)
    firstPeak = binCenter[np.argmax(pairCorr)]
    print("First peak of pair corr is at distance:", firstPeak, "equal to", firstPeak/meanRad, "times the mean radius:", meanRad)
    if(plot == True):
        plotCorrelation(binCenter, pairCorr, "$Pair$ $correlation$ $function,$ $g(r)$")
    else:
        return firstPeak

########################## Particle Self Correlations ##########################
def computeParticleVelCorr(dirName, maxPower):
    numParticles = readFromParams(dirName, "numParticles")
    timeStep = readFromParams(dirName, "dt")
    particleVelCorr = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pVel0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleVel.dat"))
    pVel0Norm = np.linalg.norm(pVel0, axis=1)**2
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(0,stepRange.shape[0]):
        pVel = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particleVel.dat"))
        particleVelCorr.append(np.mean(np.sum(np.multiply(pVel, pVel0), axis=1)/pVel0Norm))
    particleVelCorr = np.array(particleVelCorr)
    #stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "vel-corr.dat", np.column_stack((stepRange, particleVelCorr)))
    plotCorrelation((stepRange + 1) * timeStep, particleVelCorr, "$\\frac{\\langle \\vec{v}(t) \\cdot \\vec{v}(0) \\rangle}{\\langle | \\vec{v}(0) |^2 \\rangle}$", "$Simulation$ $time$", logx = True, color='k')

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeParticleLogVelCorr(dirName, startBlock, maxPower, freqPower):
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "/particleRad.dat")))
    phi = readFromParams(dirName, "phi")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    pWaveVector = np.pi /computePairCorr(dirName, plot=False)
    print("wave vector: ", pWaveVector)
    particleVelCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            stepParticleVelCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        pDir1, pDir2 = readDirectorPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pVel1, pVel2 = readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleVelCorr.append([np.mean(np.sum(np.multiply(pVel1, pVel2), axis=1)), np.mean(np.sum(np.multiply(pDir1, pDir2), axis=1))])
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleVelCorr.append(np.mean(stepParticleVelCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleVelCorr = np.array(particleVelCorr).reshape((stepList.shape[0],2))
    particleVelCorr = particleVelCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "vel-corr.dat", np.column_stack((stepList, particleVelCorr)))
    plotCorrelation(stepList, particleVelCorr[:,1], "$\\langle \\hat{n}(t) \\cdot \\hat{n}(t') \\rangle$", "$time$ $interval,$ $\\Delta t = t - t'$", logx = True, color = 'g')

########################## Particle Self Correlations ##########################
def computeParticleSusceptibility(dirName, maxPower):
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    timeStep = readFromParams(dirName, "dt")
    particleChi = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pField = np.array(np.loadtxt(dirName + os.sep + "externalField.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleChi.append(computeSusceptibility(pPos, pPos0, pField, pRad**2))
    particleChi = np.array(particleChi)
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "susceptibility.dat", np.column_stack((stepRange * timeStep, particleChi)))
    plotCorrelation(stepRange * timeStep, particleChi / (stepRange * timeStep), "$\\chi / t$", "$Simulation$ $step$", logx = True, color='k')

########################## Particle Self Correlations ##########################
def computeParticleSelfCorr(dirName, maxPower):
    computeFrom = 200
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = readFromParams(dirName, "phi")
    timeStep = readFromParams(dirName, "dt")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi / computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,3))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin.dat", np.column_stack((stepRange * timeStep, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-computeFrom:,0]/(2 * stepRange[-computeFrom:] * timeStep)), np.var(particleCorr[-computeFrom:,0]/(2 * stepRange[-computeFrom:] * timeStep)))
    #plotCorrelation(stepRange * timeStep, particleCorr[:,0]/(stepRange * timeStep), "$MSD/t$", "$Simulation$ $time,$ $t$", logx = True, logy = True, color='k')
    plotCorrelation(stepRange * timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color='k')

########## Check Self Correlations by logarithmically spaced blocks ############
def checkParticleSelfCorr(dirName, numBlocks, maxPower, plot="plot", computeTau="tau"):
    colorList = cm.get_cmap('viridis', 10)
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = readFromParams(dirName, "phi")
    timeStep = readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    print(timeStep)
    #pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi /computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    tau = []
    diff = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    start = np.argwhere(stepRange==0)[0,0]
    decade = int(10**(maxPower-1))
    for block in np.linspace(1, numBlocks, numBlocks, dtype=int):
        particleCorr = []
        pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/particlePos.dat"))
        pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/particleRad.dat")))
        end = np.argwhere(stepRange==block*decade)[0,0]
        #print((block-1)*decade, start, block*decade, end)
        stepBlock = stepRange[start:end]
        #print(stepBlock)
        start = end
        for i in range(1,stepBlock.shape[0]):
            pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepBlock[i]) + "/particlePos.dat"))
            particleCorr.append(computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
        particleCorr = np.array(particleCorr).reshape((stepBlock.shape[0]-1,3))
        stepBlock = stepBlock[1:]-(block-1)*decade#discard initial time
        if(plot=="plot"):
            plotCorrelation(stepBlock*timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color=colorList(block/10), show=False)
            plt.pause(0.2)
        if(computeTau=="tau"):
            diff.append(particleCorr[-10,0]/4*stepBlock[-10])
            ISF = particleCorr[:,1]
            step = stepBlock
            relStep = np.argwhere(ISF>np.exp(-1))[-1,0]
            if(relStep + 1 < step.shape[0]):
                t1 = step[relStep]
                t2 = step[relStep+1]
                ISF1 = ISF[relStep]
                ISF2 = ISF[relStep+1]
                slope = (ISF2 - ISF1)/(t2 - t1)
                intercept = ISF2 - slope * t2
                tau.append(timeStep*(np.exp(-1) - intercept)/slope)
            else:
                tau.append(timeStep*step[relStep])
    if(computeTau=="tau"):
        print("relaxation time: ", np.mean(tau), " +- ", np.std(tau))
        with open(dirName + "../tauDiff.dat", "ab") as f:
            np.savetxt(f, np.array([[timeStep, phi, T, np.mean(tau), np.std(tau), np.mean(diff), np.std(diff)]]))

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac = 1, computeTau = "tau"):
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = readFromParams(dirName, "phi")
    timeStep = readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    #pWaveVector = 2 * np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = 2 * np.pi / computePairCorr(dirName, plot=False)
    pWaveVector = 2 * np.pi / (float(qFrac) * 2 * pRad)
    print("wave vector: ", pWaveVector)
    particleCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepParticleCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],3))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log-q" + str(qFrac) + ".dat", np.column_stack((stepList, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-1:,0]/(2 * stepRange[-1:] * timeStep)))
    #plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    plotCorrelation(stepList * timeStep, particleCorr[:,1], "$ISF(t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    if(computeTau=="tau"):
        diff = np.mean(particleCorr[-1:,0]/(2 * stepRange[-1:] * timeStep))
        ISF = particleCorr[:,1]
        step = stepList
        relStep = np.argwhere(ISF>np.exp(-1))[-1,0]
        if(relStep + 1 < step.shape[0]):
            t1 = step[relStep]
            t2 = step[relStep+1]
            ISF1 = ISF[relStep]
            ISF2 = ISF[relStep+1]
            slope = (ISF2 - ISF1)/(t2 - t1)
            intercept = ISF2 - slope * t2
            tau = timeStep*(np.exp(-1) - intercept)/slope
            print("relaxation time: ", tau)
        else:
            tau = 0
            print("not enough data to compute relaxation time")
        with open(dirName + "../tauDiff.dat", "ab") as f:
            np.savetxt(f, np.array([[timeStep, pWaveVector, phi, T, tau, diff]]))

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower):
    numParticles = readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = readFromParams(dirName, "phi")
    timeStep = readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepParticleCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2, oneDim = True))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],3))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log-xdim.dat", np.column_stack((stepList, particleCorr)))
    print("diffusivity on x: ", particleCorr[-1:,0]/(2 * stepList[-1:] * timeStep))
    #plotCorrelation(stepList * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color = 'r')
    plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color = 'r')
    #plotCorrelation(stepList * timeStep, particleCorr[:,1], "$ISF(t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

############################### Self Correlations ##############################
def computeSelfCorr(dirName, maxPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    numVertices = np.sum(nv)
    phi = readFromParams(dirName, "phi")
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "radii.dat"))
    pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    waveVector = np.pi / meanRad
    particleCorr = []
    vertexCorr = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/pos.dat"))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/positions.dat"))
        particleCorr.append(computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, boxSize))
        vertexCorr.append(computeCorrFunctions(pos, pos0, boxSize, waveVector, boxSize))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,3))
    vertexCorr = np.array(vertexCorr).reshape((stepRange.shape[0]-1,3))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin.dat", np.column_stack((stepRange, particleCorr, vertexCorr)))
    plotCorrelation(stepRange, particleCorr[:,1], "$ISF$", "$Simulation$ $step$", logx = True, color='k')

########## Check Self Correlations by logarithmically spaced blocks ############
def checkSelfCorr(dirName, numBlocks, maxPower):
    colorList = cm.get_cmap('viridis', 10)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    numVertices = np.sum(nv)
    phi = readFromParams(dirName, "phi")
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "radii.dat"))
    pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    waveVector = np.pi / meanRad
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    start = np.argwhere(stepRange==0)[0,0]
    decade = int(10**(maxPower-1))
    for block in np.linspace(1, numBlocks, numBlocks, dtype=int):
        particleCorr = []
        vertexCorr = []
        pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/particlePos.dat"))
        pos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/positions.dat"))
        end = np.argwhere(stepRange==block*decade)[0,0]
        print((block-1)*decade, start, block*decade, end)
        stepBlock = stepRange[start:end]
        print(stepBlock)
        start = end
        for i in range(1,stepBlock.shape[0]):
            pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepBlock[i]) + "/particlePos.dat"))
            pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepBlock[i]) + "/positions.dat"))
            particleCorr.append(computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, boxSize))
            vertexCorr.append(computeCorrFunctions(pos, pos0, boxSize, waveVector, boxSize))
        particleCorr = np.array(particleCorr).reshape((stepBlock.shape[0]-1,3))
        vertexCorr = np.array(vertexCorr).reshape((stepBlock.shape[0]-1,3))
        stepBlock = stepBlock[1:]-(block-1)*decade#discard initial time
        plotCorrelation(stepBlock, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color=colorList(block/10), show=False)
        plt.pause(0.2)
    plt.show()

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeLogSelfCorr(dirName, startBlock, maxPower, freqPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    phi = readFromParams(dirName, "phi")
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "radii.dat"))
    pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    waveVector = np.pi / meanRad
    particleCorr = []
    vertexCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepParticleCorr = []
            stepVertexCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pos1, pPos2, pos2 = readPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, boxSize))
                        stepVertexCorr.append(computeCorrFunctions(pos1, pos2, boxSize, waveVector, boxSize))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
                vertexCorr.append(np.mean(stepVertexCorr, axis=0))
                #print(stepList[-1], np.var(stepParticleCorr, axis=0))
        print(power, stepList[-1])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],3))
    vertexCorr = np.array(vertexCorr).reshape((stepList.shape[0],3))
    particleCorr = particleCorr[np.argsort(stepList)]
    vertexCorr = vertexCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log.dat", np.column_stack((stepList, particleCorr, vertexCorr)))
    plotCorrelation(stepList, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

############# Time-averaged Self Correlations at fixed time window #############
def computeBlockSelfCorr(dirName, startBlock, maxPower, freqPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    phi = readFromParams(dirName, "phi")
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "radii.dat"))
    pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    waveVector = np.pi / meanRad
    particleCorr = []
    vertexCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    numBlocks = int(10**(maxPower-freqPower))
    decadeList = np.geomspace(10,10**maxPower,maxPower,dtype=int)
    for decade in decadeList:
        stepRange = np.linspace(0,decade,11,dtype=int)[:-1]
        #print(stepRange, decade)
        for i in range(1,stepRange.shape[0]):
            stepParticleCorr = []
            stepVertexCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                if(checkPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i])):
                    #print(multiple*freqDecade, multiple*freqDecade + stepRange[i])
                    pPos1, pos1, pPos2, pos2 = readPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i])
                    stepParticleCorr.append(computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, boxSize))
                    stepVertexCorr.append(computeCorrFunctions(pos1, pos2, boxSize, waveVector, boxSize))
                    numPairs += 1
            if(numPairs > 0):
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
                vertexCorr.append(np.mean(stepVertexCorr, axis=0))
                stepList.append(stepRange[i])
                #print(stepList[-1], np.var(stepParticleCorr, axis=0))
        print(decade, stepList[-1])
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],3))
    vertexCorr = np.array(vertexCorr).reshape((stepList.shape[0],3))
    particleCorr = particleCorr[np.argsort(stepList)]
    vertexCorr = vertexCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log.dat", np.column_stack((stepList, particleCorr, vertexCorr)))
    plotCorrelation(stepList, particleCorr[:,1], "$Mean$ $squared$ $displacement,$ $ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

############################## Shape Correlations ##############################
def computeShapeCorr(dirName, maxPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    shapeCorr = []
    # get trajectory directories
    stepRange = getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    stepRange = stepRange[stepRange<int(10**maxPower)]
    # initial shape parameter
    shape0 = readShape(dirName + os.sep + "t" + str(stepRange[0]), boxSize, nv)
    for i in range(1,stepRange.shape[0]):
        shape = readShape(dirName + os.sep + "t" + str(stepRange[i]), boxSize, nv)
        shapeCorr.append(computeShapeCorrFunction(shape0, shape))
    shapeCorr = np.array(shapeCorr)
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-shape.dat", np.column_stack((stepRange, shapeCorr)))
    plotCorrelation(stepRange, shapeCorr, "$shape$ $correlation$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########## Time-averaged Shape Correlations in log-spaced time window ##########
def computeLogShapeCorr(dirName, startBlock, maxPower, freqPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    shapeCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepShapeCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        shape1, shape2 = readShapePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])#, boxSize, nv)
                        stepShapeCorr.append(computeShapeCorrFunction(shape1, shape2))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                shapeCorr.append(np.mean(stepShapeCorr))
        print(power, stepList[-1])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    shapeCorr = np.array(shapeCorr)
    shapeCorr = shapeCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-shape.dat", np.column_stack((stepList, shapeCorr)))
    plotCorrelation(stepList, shapeCorr, "$shape$ $correlation$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

######### Time-averaged Velocity Correlation in log-spaced time window #########
def computeLogVelCorr(dirName, startBlock, maxPower, freqPower):
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    velCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepVelCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pVel1, pVel2 = readVelPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i], nv)
                        stepVelCorr.append(computeVelCorrFunction(vel1, vel2))
                        numPairs += 1
            if(numPairs > 0):
                velCorr.append(np.mean(stepVelCorr))
                stepList.append(stepRange[i])
        print(decade, stepList[-1])
    stepList = np.array(stepList)
    velCorr = np.array(velCorr)
    velCorr = velCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-vel.dat", np.column_stack((stepList, velCorr)))
    plotCorrelation(stepList, velCorr, "$velocity$ $correlation$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########################### Hexitic Order Parameter ############################
def computeVelCorrContact(dirName, nv):
    numParticles = nv.shape[0]
    vel = np.array(np.loadtxt(dirName + os.sep + "velocities.dat"))
    pVel = computeParticleVelocities(vel, nv)
    meanVel = np.linalg.norm(np.mean(pVel, axis=0))
    contacts = np.array(np.loadtxt(dirName + os.sep + "neighbors.dat"), dtype = int)
    velcontact = np.zeros(numParticles)
    for i in range(numParticles):
        numContacts = 0
        for c in range(contacts[i].shape[0]):
            if(contacts[i,c] != -1):
                numContacts += 1
                velcontact[i] += np.dot(pVel[i]/np.linalg.norm(pVel[i]), pVel[contacts[i,c]]/np.linalg.norm(pVel[contacts[i,c]]))
        if(numContacts > 0):
            velcontact[i] /= numContacts
    return velcontact

def computeVelCorrDistance(dirName, boxSize, nv, distanceTh = 0.1):
    #distanceTh *= boxSize[0]
    numParticles = nv.shape[0]
    vel = np.array(np.loadtxt(dirName + os.sep + "velocities.dat"))
    pVel = computeParticleVelocities(vel, nv)
    meanVel = np.linalg.norm(np.mean(pVel, axis=0))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = computeDistances(pPos, boxSize) / boxSize[0]
    veldistance = np.zeros(numParticles)
    for i in range(numParticles):
        distList = np.argwhere(distance[i]<distanceTh)[:,0]
        for j in distList:
            if(distance[i,j] > 0):
                veldistance[i] += np.dot(pVel[i]/np.linalg.norm(pVel[i]), pVel[j]/np.linalg.norm(pVel[j]))
        veldistance[i] /= distList.shape[0]-1
    return veldistance

def computeLocalAreaGrid(pos, area, xbin, ybin, localArea):
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localArea[x, y] += area[pId]

############################ Local Packing Fraction ############################
def computeLocalDensity(dirName, numBins, plot = False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(readFromParams(dirName, "numParticles"))
    area = np.array(np.loadtxt(dirName + os.sep + "restAreas.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    print("subBoxSize: ", xbin[1]-xbin[0], ybin[1]-ybin[0], " lengthscale: ", np.sqrt(np.mean(area)))
    localArea = np.zeros((numBins, numBins))
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    computeLocalAreaGrid(pos, area, xbin, ybin, localArea)
    localDensity = localArea/localSquare
    localDensity = np.sort(localDensity.flatten())
    cdf = np.arange(len(localDensity))/len(localDensity)
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 30), density=True)
    edges = (edges[1:] + edges[:-1])/2
    print("data stats: ", np.min(localDensity), np.max(localDensity), np.mean(localDensity), np.std(localDensity))
    if(plot=="plot"):
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        ax.plot(edges[1:], pdf[1:], linewidth=1.2, color='k')
        #ax.plot(localDensity, cdf, linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel('$P(\\varphi)$', fontsize=18)
        ax.set_xlabel('$\\varphi$', fontsize=18)
        #ax.set_xlim(-0.02, 1.02)
        plt.tight_layout()
        plt.show()

############################ Local Packing Fraction ############################
def averageLocalDensity(dirName, numBins, plot = False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(readFromParams(dirName, "numParticles"))
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    sampleDensity = []
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    numSamples = 0
    for dir in os.listdir(dirName):
        if(os.path.exists(dirName + os.sep + dir + os.sep + "restAreas.dat")):
            localArea = np.zeros((numBins, numBins))
            area = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "restAreas.dat"))
            pPos = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "particlePos.dat"))
            pPos[:,0] -= np.floor(pPos[:,0]/boxSize[0]) * boxSize[0]
            pPos[:,1] -= np.floor(pPos[:,1]/boxSize[1]) * boxSize[1]
            computeLocalAreaGrid(pPos, area, xbin, ybin, localArea)
            localDensity = localArea/localSquare
            sampleDensity.append(localDensity.flatten())
    sampleDensity = np.sort(sampleDensity)
    cdf = np.arange(len(sampleDensity))/len(sampleDensity)
    pdf, edges = np.histogram(sampleDensity, bins=np.linspace(np.min(sampleDensity), np.max(sampleDensity), 30), density=True)
    edges = (edges[:-1] + edges[1:])/2
    print("data stats: ", np.min(sampleDensity), np.max(sampleDensity), np.mean(sampleDensity), np.std(sampleDensity))
    if(plot=="plot"):
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        ax.plot(edges[1:], pdf[1:], linewidth=1.2, color='k')
        #ax.plot(sampleDensity, cdf, linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel('$P(\\varphi)$', fontsize=18)
        ax.set_xlabel('$\\varphi$', fontsize=18)
        #ax.set_xlim(-0.02, 1.02)
        plt.tight_layout()
        plt.pause(0.5)
    np.savetxt(dirName + "localDensity-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))

########################### Hexitic Order Parameter ############################
def computeHexaticOrder(dirName, boxSize):
    numParticles = int(readFromParams(dirName, "numParticles"))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "contacts.dat"), dtype = int)
    psi6 = np.zeros(numParticles)
    for i in range(numParticles):
        numContacts = 0
        for c in range(contacts[i].shape[0]):
            if(contacts[i,c] != -1):
                numContacts += 1
                delta = pbcDistance(pPos[i], pPos[contacts[i,c]], boxSize)
                theta = np.arctan2(delta[1], delta[0])
                psi6[i] += np.exp(6j*theta)
        if(numContacts > 0):
            psi6[i] /= numContacts
            psi6[i] = np.abs(psi6[i])
    return psi6

########################## Hexitic Order Correlation ###########################
def computeHexaticCorrelation(dirName, boxSize):
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    psi6 = computeHexaticOrder(dirName)
    distance = computeDistances(pPos, boxSize) / boxSize[0]
    bins = np.linspace(np.min(distance[distance>0]), np.max(distance), 50)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    hexCorr = np.zeros(binCenter.shape[0])
    counts = np.zeros(binCenter.shape[0])
    for i in range(1,pPos.shape[0]):
        for j in range(i):
            for k in range(bins.shape[0]-1):
                if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                    hexCorr[k] += psi6[i] * np.conj(psi6[j])
                    counts[k] += 1
    hexCorr /= counts
    return binCenter, hexCorr

############################# Velocity Correlation #############################
def computeVelocityHistogram(dirName, boxSize, nv, numBins):
    numParticles = nv.shape[0]
    vel = np.array(np.loadtxt(dirName + os.sep + "velocities.dat"))
    pVel = np.zeros((numParticles, 2))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = computeDistances(pPos, boxSize) / boxSize[0] # only works for square box
    bins = np.linspace(np.min(distance[distance>0]), np.max(distance), numBins)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    velCorr = []
    firstVertex = 0
    for pId in range(numParticles):
        idList = np.arange(firstVertex, firstVertex+nv[pId], 1)
        pVel[pId] = [np.mean(vel[firstVertex:firstVertex+nv[pId],0]), np.mean(vel[firstVertex:firstVertex+nv[pId],1])]
        firstVertex += nv[pId]
    for i in range(1, numParticles):
        pvelcorr = np.zeros(numBins-1)
        pcounts = np.zeros(numBins-1)
        for j in range(i):
            for k in range(numBins-1):
                if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                    pvelcorr[k] += np.dot(pVel[i]/np.linalg.norm(pVel[i]), pVel[j]/np.linalg.norm(pVel[j]))
                    pcounts[k] += 1
        pvelcorr[pcounts>0] /= pcounts[pcounts>0]
        velCorr.append(pvelcorr)
    velCorr = np.array(velCorr)
    velCorr = np.mean(velCorr, axis=0)
    return binCenter, velCorr


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "paircorr"):
        computePairCorr(dirName)

    elif(whichCorr == "pvel"):
        maxPower = int(sys.argv[3])
        computeParticleVelCorr(dirName, maxPower)

    elif(whichCorr == "plogvel"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogVelCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "psus"):
        maxPower = int(sys.argv[3])
        computeParticleSusceptibility(dirName, maxPower)

    elif(whichCorr == "pcorr"):
        maxPower = int(sys.argv[3])
        computeParticleSelfCorr(dirName, maxPower)

    elif(whichCorr == "pcheckcorr"):
        numBlocks = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        plot = sys.argv[5]
        computeTau = sys.argv[6]
        checkParticleSelfCorr(dirName, numBlocks, maxPower, plot=plot, computeTau=computeTau)

    elif(whichCorr == "plogcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        qFrac = sys.argv[6]
        computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac)

    elif(whichCorr == "plogcorrx"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "density"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensity(dirName, numBins, plot)

    elif(whichCorr == "averagedensity"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        averageLocalDensity(dirName, numBins, plot)

    elif(whichCorr == "lincorr"):
        maxPower = int(sys.argv[3])
        computeSelfCorr(dirName, maxPower)

    elif(whichCorr == "checkcorr"):
        numBlocks = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        checkSelfCorr(dirName, numBlocks, maxPower)

    elif(whichCorr == "logcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeLogSelfCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "blockcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeBlockSelfCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "linshape"):
        maxPower = int(sys.argv[3])
        computeShapeCorr(dirName, maxPower)

    elif(whichCorr == "logshape"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeLogShapeCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "logvel"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeLogVelCorr(dirName, startBlock, maxPower, freqPower)

    else:
        print("Please specify the correlation you want to compute")
