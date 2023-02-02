'''
Created by Francesco
14 September 2022
'''
#these functions are for computng correlation functions without plotting on the cluster
import numpy as np
import utilsCorr as ucorr
import sys
import os

########################### Pair Correlation Function ##########################
def computePairCorr(dirName):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    meanRad = np.mean(rad)
    pos = np.loadtxt(dirName + os.sep + "particlePos.dat")
    pos = np.array(pos)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    distance = ucorr.computeDistances(pos[rad>np.mean(rad)], boxSize).flatten()
    distance = distance[distance>0]
    bins = np.linspace(np.min(distance), np.max(distance), 50)
    pairCorr, edges = np.histogram(distance, bins=bins, density=True)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    pairCorr /= (phi * 2 * np.pi * binCenter)
    firstPeak = binCenter[np.argmax(pairCorr)]
    print("First peak of pair corr is at distance:", firstPeak, "equal to", firstPeak/meanRad, "times the mean radius:", meanRad)
    return firstPeak

###################### One Dim Particle Self Correlations ######################
def computeParticleSelfCorrOneDim(dirName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2, oneDim = True))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin-xdim.dat", np.column_stack((stepRange * timeStep, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)))

########### One Dim Time-averaged Self Corr in log-spaced time window ##########
def computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(ucorr.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2, oneDim = True))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],7))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log-xdim.dat", np.column_stack((stepList * timeStep, particleCorr)))
    print("diffusivity on x: ", np.mean(particleCorr[-20:,0]/(2*stepList[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(2*stepList[-20:]*timeStep)))

############################ Particle Susceptibility ###########################
def computeParticleSusceptibility(dirName, sampleName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleChi = []
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pos0 = np.array(np.loadtxt(dirName + os.sep + "../" + sampleName + "/t" + str(stepRange[0]) + "/particlePos.dat"))
    pField = np.array(np.loadtxt(dirName + os.sep + "externalField.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    pWaveVector = np.pi / pRad
    damping = 1e03
    scale = pRad**2
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "../" + sampleName + "/t" + str(stepRange[i]) + "/particlePos.dat"))
        particleChi.append(ucorr.computeSusceptibility(pPos, pPos0, pField, pWaveVector, scale))
        #particleChi.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, scale, oneDim = True))
        particleCorr.append(ucorr.computeCorrFunctions(pos, pos0, boxSize, pWaveVector, scale, oneDim = True))
    particleChi = np.array(particleChi)
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "sus-lin-xdim.dat", np.column_stack((stepRange*timeStep, particleChi)))
    np.savetxt(dirName + os.sep + "../dynamics-test/corr-lin-xdim.dat", np.column_stack((stepRange*timeStep, particleCorr)))
    print("susceptibility: ", np.mean(particleChi[-20:,0]/(stepRange[-20:]*timeStep)), " ", np.std(particleChi[-20:,0]/(stepRange[-20:]*timeStep)))

######################### Tracer Velocity Distribution #########################
def computeParticleVelPDFSubSet(dirName, firstIndex=10, mass=1e06):
    vel = []
    velSubSet = []
    step = []
    numParticles = ucorr.readFromParams(dirName + os.sep + "t0", "numParticles")
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            pVel = np.loadtxt(dirName + os.sep + dir + os.sep + "particleVel.dat")
            vel.append(pVel[firstIndex:,:])
            subset = pVel[:firstIndex,:] * np.sqrt(mass)
            velSubSet.append(subset)
            step.append(float(dir[1:]))
    vel = np.array(vel).flatten()
    velSubSet = np.array(velSubSet).flatten()
    step = np.sort(step)
    #velSubSet /= np.sqrt(2*np.var(velSubSet))
    velBins = np.linspace(np.min(velSubSet), np.max(velSubSet), 50)
    velPDF, edges = np.histogram(vel, bins=velBins, density=True)
    velSubSetPDF, edges = np.histogram(velSubSet, bins=velBins, density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(dirName + os.sep + "velocityPDF.dat", np.column_stack((edges, velPDF, velSubSetPDF)))
    np.savetxt(dirName + os.sep + "tracerTemp.dat", np.column_stack((np.var(vel), np.var(velSubSet))))
    print("Variance of the velocity pdf:", np.var(vel), " variance of the subset velocity pdf: ", np.var(velSubSet))

########################### Average Space Correlator ###########################
def averagePairCorr(dirName, dirSpacing=1000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = ucorr.readFromParams(dirName, "phi")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    minRad = np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    rbins = np.arange(0, np.sqrt(2)*boxSize[0]/2, 0.05*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    print(dirList.shape[0])
    pcorr = np.zeros(rbins.shape[0]-1)
    for dir in dirList:
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        pcorr += ucorr.getPairCorr(pos, boxSize, rbins, minRad)/(numParticles * phi)
    pcorr[pcorr>0] /= dirList.shape[0]
    binCenter = (rbins[:-1] + rbins[1:])*0.5
    firstPeak = binCenter[np.argmax(pcorr)]
    np.savetxt(dirName + os.sep + "pairCorr.dat", np.column_stack((binCenter, pcorr)))
    print("First peak of pair corr is at distance:", firstPeak, "equal to", firstPeak/minRad, "times the min radius:", minRad)
    np.savetxt(dirName + os.sep + "pcorrFirstPeak.dat", np.column_stack((firstPeak, np.max(pcorr))))

########################## Local Density Distribution ##########################
def averageLocalDensity(dirName, numBins=12):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    localDensity = []
    for dir in dirList:
        localArea = np.zeros((numBins, numBins))
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        ucorr.computeLocalAreaGrid(pos, rad, xbin, ybin, localArea)
        localDensity.append(localArea/localSquare)
    localDensity = np.array(localDensity).flatten()
    localDensity = np.sort(localDensity)
    localDensity = localDensity[localDensity > 0]
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 50), density=True)
    edges = (edges[:-1] + edges[1:])/2
    np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))
    alpha2 = np.mean(localDensity**4)/(2*np.mean(localDensity**2)**2) - 1
    np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + "-stats.dat", np.column_stack((np.mean(localDensity), np.var(localDensity), alpha2)))

############################ Velocity distribution #############################
def averageParticleVelPDFCluster(dirName, dirSpacing=1000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    dirList = dirList[-50:]
    velInCluster = np.empty(0)
    velOutCluster = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "clusterList.dat")):
            clusterList = np.loadtxt(dirSample + os.sep + "clusterList.dat")
            inLabel = clusterList[:,1]
            outLabel = clusterList[:,1]
        else:
            inLabel = searchClusters(dirSample)
            outLabel = np.loadtxt(dirSample + os.sep + "clusterList.dat")[:,2]
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        velNorm = np.linalg.norm(vel, axis=1)
        velInCluster = np.append(velInCluster, velNorm[inLabel==1].flatten())
        velOutCluster = np.append(velOutCluster, velNorm[outLabel==1].flatten())
    # in cluster
    mean = np.mean(velInCluster)
    Temp = np.var(velInCluster)
    skewness = np.mean((velInCluster - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velInCluster - mean)**4)/Temp**2
    data = velInCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity in cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    np.savetxt(dirName + os.sep + "velPDFInCluster.dat", np.column_stack((edges, pdf)))
    # out of cluster
    mean = np.mean(velOutCluster)
    Temp = np.var(velOutCluster)
    skewness = np.mean((velOutCluster - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velOutCluster - mean)**4)/Temp**2
    data = velOutCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity out cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    np.savetxt(dirName + os.sep + "velPDFOutCluster.dat", np.column_stack((edges, pdf)))

########################### Average Space Correlator ###########################
def getCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    if(os.path.exists(dirName + "/collisionIntervals.dat") and check=="check"):
        interval = np.loadtxt(dirName + os.sep + "collisionIntervals.dat")
    else:
        interval = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousVel = np.array(np.loadtxt(dirName + os.sep + "t0/particleVel.dat"))
        for i in range(1,dirList.shape[0]):
            currentTime = timeList[i]
            currentVel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
            colIndex = np.argwhere(currentVel[:,0]!=previousVel[:,0])[:,0]
            currentInterval = currentTime-previousTime[colIndex]
            interval = np.append(interval, currentInterval[currentInterval>1])
            previousTime[colIndex] = currentTime
            previousVel = currentVel
        interval = np.sort(interval)
        interval *= timeStep
        np.savetxt(dirName + os.sep + "collisionIntervals.dat", interval)
    bins = np.linspace(np.min(interval), np.max(interval), numBins)
    pdf, edges = np.histogram(interval, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time:", np.mean(interval), " standard deviation: ", np.std(interval))
    np.savetxt(dirName + os.sep + "collision.dat", np.column_stack((centers, pdf)))

########################### Average Space Correlator ###########################
def getContactCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    if(os.path.exists(dirName + "/contactCollisionIntervals.dat") and check=="check"):
        interval = np.loadtxt(dirName + os.sep + "contactCollisionIntervals.dat")
    else:
        interval = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousContacts = np.array(np.loadtxt(dirName + os.sep + "t0/particleContacts.dat"))
        for i in range(1,dirList.shape[0]):
            currentTime = timeList[i]
            currentContacts = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleContacts.dat"), dtype=np.int64)
            colIndex = np.argwhere(currentContacts!=previousContacts)[:,0]
            currentInterval = currentTime-previousTime[colIndex]
            interval = np.append(interval, currentInterval[currentInterval>1])
            previousTime[colIndex] = currentTime
            previousContacts = currentContacts
        interval = np.sort(interval)
        interval *= timeStep
        np.savetxt(dirName + os.sep + "contactCollisionIntervals.dat", interval)
    bins = np.arange(np.min(interval), np.max(interval), 5*np.min(interval))
    pdf, edges = np.histogram(interval, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time:", np.mean(interval), " standard deviation: ", np.std(interval))
    np.savetxt(dirName + os.sep + "contactCollision.dat", np.column_stack((centers, pdf)))

################# Cluster contact rearrangement distribution ###################
def getClusterContactCollisionIntervalPDF(dirName, check=False, numBins=40, cluster="cluster"):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    dirSpacing = 100
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    if(os.path.exists(dirName + "/contactCollisionIntervals.dat") and check=="check"):
        print("loading already existing file")
        interval = np.loadtxt(dirName + os.sep + "contactCollisionIntervals.dat")
    else:
        interval = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousContacts = np.array(np.loadtxt(dirName + os.sep + "t0/particleContacts.dat"))
        for i in range(1,dirList.shape[0]):
            dirSample = dirName + os.sep + dirList[i]
            if(os.path.exists(dirSample + os.sep + "clusterList.dat")):
                if(cluster == "nocluster"):
                    clusterList = np.loadtxt(dirSample + os.sep + "clusterList.dat")[:,2]
                else:
                    clusterList = np.loadtxt(dirSample + os.sep + "clusterList.dat")[:,1]
            else:
                clusterList = searchClusters(dirSample)
            particlesInClusterIndex = np.argwhere(clusterList==1)[:,0]
            currentTime = timeList[i]
            currentContacts = np.array(np.loadtxt(dirSample + "/particleContacts.dat"), dtype=np.int64)
            colIndex = np.unique(np.argwhere(currentContacts!=previousContacts)[:,0])
            colIndex = np.intersect1d(colIndex, particlesInClusterIndex)
            currentInterval = currentTime-previousTime[colIndex]
            interval = np.append(interval, currentInterval[currentInterval>1])
            previousTime[colIndex] = currentTime
            previousContacts = currentContacts
        interval = np.sort(interval)
        interval *= timeStep
        np.savetxt(dirName + os.sep + "clusterCollisionIntervals.dat", interval)
    #bins = np.arange(np.min(interval), np.max(interval), spacing*timeStep)
    bins = np.arange(np.min(interval), np.max(interval), 5*np.min(interval))
    pdf, edges = np.histogram(interval, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time:", np.mean(interval), " standard deviation: ", np.std(interval))
    np.savetxt(dirName + os.sep + "clusterCollision.dat", np.column_stack((centers, pdf)))

########################## Particle Self Correlations ##########################
def computeParticleSelfCorr(dirName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "linCorr.dat", np.column_stack((stepRange*timeStep, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)))

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac = 1, computeTau = "tau"):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    if(os.path.exists(dirName + os.sep + "pairCorr.dat")):
        pcorr = np.loadtxt(dirName + os.sep + "pairCorr.dat")
        firstPeak = pcorr[np.argmax(pcorr[:,1]),0]
        pWaveVector = 2 * np.pi / firstPeak
    else:
        pWaveVector = 2 * np.pi / (float(qFrac) * 2 * pRad)
    print("wave vector: ", pWaveVector, " meanRad: ", pRad)
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(ucorr.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],7))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "logCorr-q" + str(qFrac) + ".dat", np.column_stack((stepList, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(4*stepList[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(4*stepList[-20:]*timeStep)))
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
        with open(dirName + "../../../tauDiff.dat", "ab") as f:
            np.savetxt(f, np.array([[timeStep, pWaveVector, phi, T, tau, diff]]))

##################### Particle Self Velocity Correlations ######################
def computeParticleVelCorr(dirName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleVelCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pVel0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleVel.dat"))
    pVel0Norm = np.linalg.norm(pVel0, axis=1)**2
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(0,stepRange.shape[0]):
        pVel = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particleVel.dat"))
        particleVelCorr.append(np.mean(np.sum(np.multiply(pVel, pVel0), axis=1)/pVel0Norm))
    particleVelCorr = np.array(particleVelCorr)
    np.savetxt(dirName + os.sep + "velCorr.dat", np.column_stack((stepRange, particleVelCorr)))

############# Time-averaged Self Vel Corr in log-spaced time window ############
def computeParticleLogVelCorr(dirName, startBlock, maxPower, freqPower):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleVelCorr = []
    particleDirCorr = []
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        pVel1, pVel2 = ucorr.readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pDir1, pDir2 = ucorr.readDirectorPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleVelCorr.append(np.mean(np.sum(np.multiply(pVel1,pVel2), axis=1)))
                        stepParticleDirCorr.append(np.mean(np.sum(np.multiply(pDir1,pDir2), axis=1)))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleVelCorr.append([np.mean(stepParticleVelCorr, axis=0), np.std(stepParticleVelCorr, axis=0)])
                particleDirCorr.append([np.mean(stepParticleDirCorr, axis=0), np.std(stepParticleDirCorr, axis=0)])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleVelCorr = np.array(particleVelCorr).reshape((stepList.shape[0],2))
    particleVelCorr = particleVelCorr[np.argsort(stepList)]
    particleDirCorr = np.array(particleDirCorr).reshape((stepList.shape[0],2))
    particleDirCorr = particleDirCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "logVelCorr.dat", np.column_stack((stepList*timeStep, particleVelCorr)))
    np.savetxt(dirName + os.sep + "logDirCorr.dat", np.column_stack((stepList*timeStep, particleDirCorr)))

########################## Space Velocity Correlation ##########################
def averageParticleVelSpaceCorr(dirName, dirSpacing=1000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.arange(2*minRad, np.sqrt(2)*boxSize[0]/2, 2*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    dirList = dirList[-10:]
    velCorr = np.zeros((bins.shape[0]-1,3))
    counts = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        pos = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particlePos.dat"))
        distance = ucorr.computeDistances(pos, boxSize)
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particleVel.dat"))
        velNorm = np.linalg.norm(vel, axis=1)
        velNormSquared = np.mean(velNorm**2)
        for i in range(distance.shape[0]):
            for j in range(i):
                    for k in range(bins.shape[0]-1):
                        if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                            # parallel
                            delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                            parProj1 = np.dot(vel[i],delta)
                            parProj2 = np.dot(vel[j],delta)
                            velCorr[k,0] += parProj1 * parProj2
                            # perpendicular
                            deltaPerp = np.array([-delta[1], delta[0]])
                            perpProj1 = np.dot(vel[i],deltaPerp)
                            perpProj2 = np.dot(vel[j],deltaPerp)
                            velCorr[k,1] += perpProj1 * perpProj2
                            # total
                            velCorr[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                            counts[k] += 1
    velCorr[counts>0,0] /= counts[counts>0]
    velCorr[counts>0,1] /= counts[counts>0]
    velCorr[counts>0,2] /= counts[counts>0]
    velCorr /= velNormSquared
    binCenter = (bins[1:] + bins[:-1])/2
    np.savetxt(dirName + os.sep + "spaceVelCorr.dat", np.column_stack((binCenter, velCorr, counts)))

##################### Velocity Correlation in/out Cluster ######################
def averageParticleVelSpaceCorrCluster(dirName, dirSpacing=1000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.arange(2*minRad, np.sqrt(2)*boxSize[0]/2, 2*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    dirList = dirList[-10:]
    if(os.path.exists(dirName + os.sep + "clusterList.dat")):
        clusterList = np.loadtxt(dirSample + os.sep + "clusterList.dat")
        inLabel = clusterList[:,1]
        outLabel = clusterList[:,1]
    else:
        inLabel = searchClusters(dirName)
        outLabel = np.loadtxt(dirName + os.sep + "clusterList.dat")[:,2]
    velCorrInCluster = np.zeros((bins.shape[0]-1,3))
    countsInCluster = np.zeros(bins.shape[0]-1)
    velCorrOutCluster = np.zeros((bins.shape[0]-1,3))
    countsOutCluster = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        pos = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particlePos.dat"))
        distance = ucorr.computeDistances(pos, boxSize)
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particleVel.dat"))
        velNorm = np.linalg.norm(vel, axis=1)
        velNormSquared = np.mean(velNorm**2)
        for i in range(distance.shape[0]):
            for j in range(i):
                    for k in range(bins.shape[0]-1):
                        if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                            # parallel
                            delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                            parProj1 = np.dot(vel[i],delta)
                            parProj2 = np.dot(vel[j],delta)
                            # perpendicular
                            deltaPerp = np.array([-delta[1], delta[0]])
                            perpProj1 = np.dot(vel[i],deltaPerp)
                            perpProj2 = np.dot(vel[j],deltaPerp)
                            # correlations
                            if(inLabel[i]==1):
                                velCorrInCluster[k,0] += parProj1 * parProj2
                                velCorrInCluster[k,1] += perpProj1 * perpProj2
                                velCorrInCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                                countsInCluster[k] += 1
                            if(outLabel[i]==1):
                                velCorrOutCluster[k,0] += parProj1 * parProj2
                                velCorrOutCluster[k,1] += perpProj1 * perpProj2
                                velCorrOutCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                                countsOutCluster[k] += 1
    binCenter = (bins[1:] + bins[:-1])/2
    for i in range(velCorrInCluster.shape[1]):
        velCorrInCluster[countsOutCluster>0,i] /= countsInCluster[countsOutCluster>0]
        velCorrOutCluster[countsOutCluster>0,i] /= countsOutCluster[countsOutCluster>0]
    velCorrInCluster /= velNormSquared
    velCorrOutCluster /= velNormSquared
    np.savetxt(dirName + os.sep + "spaceVelCorrInCluster.dat", np.column_stack((binCenter, velCorrInCluster, countsInCluster)))
    np.savetxt(dirName + os.sep + "spaceVelCorrOutCluster.dat", np.column_stack((binCenter, velCorrOutCluster, countsOutCluster)))

############################# Clustering algorithm #############################
def searchClusters(dirName):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    particleLabel = np.zeros(numParticles)
    connectLabel = np.zeros(numParticles)
    clusterLabel = 0
    for i in range(1,numParticles):
        if(np.sum(contacts[i]!=-1)>2):
            if(particleLabel[i] == 0): # this means it hasn't been checked yet
                # check that it is not a contact of contacts of previously checked particles
                belongToCluster = False
                for j in range(i):
                    for c in contacts[j, np.argwhere(contacts[j]!=-1)[:,0]]:
                        if(i==c):
                            # a contact of this particle already belongs to a cluster
                            particleLabel[i] = particleLabel[j]
                            belongToCluster = True
                            break
                if(belongToCluster == False):
                    newCluster = False
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(np.sum(contacts[c]!=-1)>2 and newCluster == False):
                            newCluster = True
                    if(newCluster == True):
                        clusterLabel += 1
                        particleLabel[i] = clusterLabel
                        particleLabel[contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]] = particleLabel[i]
        else:
            particleLabel[i] = 0
    # more stringent condition on cluster belonging
    connectLabel[np.argwhere(particleLabel > 0)] = 1
    for i in range(numParticles):
        connected = False
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            if(particleLabel[c] != 0 and connected == False):
                connectLabel[i] = 1
                connected = True
    noClusterLabel = np.zeros(numParticles)
    for i in range(numParticles):
        if(particleLabel[i] == 0 and np.sum(contacts[i]) < 2):
            noClusterLabel[i] = 1
    np.savetxt(dirName + "/clusterList.dat", np.column_stack((particleLabel, connectLabel, noClusterLabel)))
    return connectLabel


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "pc"):
        computePairCorr(dirName)

############################### functions for FDT ##############################
    elif(whichCorr == "corr1d"):
        maxPower = int(sys.argv[3])
        computeParticleSelfCorrOneDim(dirName, maxPower)

    elif(whichCorr == "logcorr1d"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "sus"):
        sampleName = sys.argv[3]
        maxPower = int(sys.argv[4])
        computeParticleSusceptibility(dirName, sampleName, maxPower)

    elif(whichCorr == "pvelsubset"):
        firstIndex = int(sys.argv[3])
        mass = float(sys.argv[4])
        computeParticleVelPDFSubSet(dirName, firstIndex, mass)

########################### pos and vel correlations ###########################
    elif(whichCorr == "averagepc"):
        averagePairCorr(dirName)

    elif(whichCorr == "averageld"):
        numBins = int(sys.argv[3])
        averageLocalDensity(dirName, numBins)

    elif(whichCorr == "velpdfcluster"):
        averageParticleVelPDFCluster(dirName)

    elif(whichCorr == "collision"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getCollisionIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "contactcol"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getContactCollisionIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "clustercol"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        cluster = sys.argv[5]
        getContactCollisionIntervalPDF(dirName, check, numBins, cluster)

    elif(whichCorr == "corrsingle"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        qFrac = sys.argv[6]
        computeSingleParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac)

    elif(whichCorr == "corr"):
        maxPower = int(sys.argv[3])
        computeParticleSelfCorr(dirName, maxPower)

    elif(whichCorr == "logcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        qFrac = sys.argv[6]
        computeTau = sys.argv[7]
        computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac, computeTau=computeTau)

    elif(whichCorr == "vlogcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogVelCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "averagevc"):
        averageParticleVelSpaceCorr(dirName)

    elif(whichCorr == "averagevccluster"):
        averageParticleVelSpaceCorrCluster(dirName)

    elif(whichCorr == "cluster"):
        searchClusters(dirName)

    else:
        print("Please specify the correlation you want to compute")
