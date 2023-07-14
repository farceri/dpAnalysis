'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import utilsPlot as uplot
import sys
import os
import utils

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
    stepRange = utils.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/pos.dat"))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/positions.dat"))
        particleCorr.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, boxSize))
        vertexCorr.append(utils.computeCorrFunctions(pos, pos0, boxSize, waveVector, boxSize))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,3))
    vertexCorr = np.array(vertexCorr).reshape((stepRange.shape[0]-1,3))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin.dat", np.column_stack((stepRange, particleCorr, vertexCorr)))
    uplot.plotCorrelation(stepRange, particleCorr[:,1], "$ISF$", "$Simulation$ $step$", logx = True, color='k')

########### Plot Self Correlations by logarithmically spaced blocks ############
def plotSelfCorr(dirName, numBlocks, maxPower):
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
    stepRange = utils.getDirectories(dirName)
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
            particleCorr.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, boxSize))
            vertexCorr.append(utils.computeCorrFunctions(pos, pos0, boxSize, waveVector, boxSize))
        particleCorr = np.array(particleCorr).reshape((stepBlock.shape[0]-1,3))
        vertexCorr = np.array(vertexCorr).reshape((stepBlock.shape[0]-1,3))
        stepBlock = stepBlock[1:]-(block-1)*decade#discard initial time
        uplot.plotCorrelation(stepBlock, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color=colorList(block/10), show=False)
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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pos1, pPos2, pos2 = utils.readPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(utils.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, boxSize))
                        stepVertexCorr.append(utils.computeCorrFunctions(pos1, pos2, boxSize, waveVector, boxSize))
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
    uplot.plotCorrelation(stepList, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

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
                if(utils.checkPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i])):
                    #print(multiple*freqDecade, multiple*freqDecade + stepRange[i])
                    pPos1, pos1, pPos2, pos2 = utils.readPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i])
                    stepParticleCorr.append(utils.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, boxSize))
                    stepVertexCorr.append(utils.computeCorrFunctions(pos1, pos2, boxSize, waveVector, boxSize))
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
    uplot.plotCorrelation(stepList, particleCorr[:,1], "$Mean$ $squared$ $displacement,$ $ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

############################## Shape Correlations ##############################
def computeShapeCorr(dirName, maxPower):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype = int)
    numParticles = nv.shape[0]
    shapeCorr = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    stepRange = stepRange[stepRange<int(10**maxPower)]
    # initial shape parameter
    shape0 = shapeDescriptors.readShape(dirName + os.sep + "t" + str(stepRange[0]), boxSize, nv)
    for i in range(1,stepRange.shape[0]):
        shape = shapeDescriptors.readShape(dirName + os.sep + "t" + str(stepRange[i]), boxSize, nv)
        shapeCorr.append(utils.computeShapeCorrFunction(shape0, shape))
    shapeCorr = np.array(shapeCorr)
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-shape.dat", np.column_stack((stepRange, shapeCorr)))
    uplot.plotCorrelation(stepRange, shapeCorr, "$shape$ $correlation$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pVel1, pVel2 = utils.readVelPair(dirName, multiple*freqDecade, multiple*freqDecade + stepRange[i], nv)
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
    uplot.plotCorrelation(stepList, velCorr, "$velocity$ $correlation$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########################### Hexitic Order Parameter ############################
def computeVelCorrContact(dirName, nv):
    numParticles = nv.shape[0]
    vel = np.array(np.loadtxt(dirName + os.sep + "velocities.dat"))
    pVel = utils.computeParticleVelocities(vel, nv)
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
    pVel = utils.computeParticleVelocities(vel, nv)
    meanVel = np.linalg.norm(np.mean(pVel, axis=0))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = utils.computeDistances(pPos, boxSize) / boxSize[0]
    veldistance = np.zeros(numParticles)
    for i in range(numParticles):
        distList = np.argwhere(distance[i]<distanceTh)[:,0]
        for j in distList:
            if(distance[i,j] > 0):
                veldistance[i] += np.dot(pVel[i]/np.linalg.norm(pVel[i]), pVel[j]/np.linalg.norm(pVel[j]))
        veldistance[i] /= distList.shape[0]-1
    return veldistance


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    elif(whichCorr == "lincorr"):
        maxPower = int(sys.argv[3])
        computeSelfCorr(dirName, maxPower)

    elif(whichCorr == "plotcorr"):
        numBlocks = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        plotSelfCorr(dirName, numBlocks, maxPower)

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

    elif(whichCorr == "logvel"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeLogVelCorr(dirName, startBlock, maxPower, freqPower)

    else:
        print("Please specify the correlation you want to compute")
