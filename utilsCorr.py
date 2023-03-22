'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from scipy.fft import fft, fftfreq, fft2
from sklearn.cluster import DBSCAN
import os

############################## general utilities ###############################
def pbcDistance(r1, r2, boxSize):
    delta = r1 - r2
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

def computeDistances(pos, boxSize):
    numParticles = pos.shape[0]
    distances = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    distances += boxSize / 2
    distances %= boxSize
    distances -= boxSize / 2
    distances = np.sqrt(np.sum(distances**2, axis=2))
    return distances

def computeNeighbors(pos, boxSize, cutoff, maxNeighbors=20):
    numParticles = pos.shape[0]
    neighbors = np.ones((numParticles, maxNeighbors))*-1
    neighborCount = np.zeros(numParticles, dtype=int)
    distance = computeDistances(pos, boxSize)
    for i in range(1,numParticles):
        for j in range(i):
            if(distance[i,j] < cutoff):
                neighbors[i,neighborCount[i]] = j
                neighbors[j,neighborCount[j]] = i
                neighborCount[i] += 1
                neighborCount[j] += 1
                if(neighborCount[i] > maxNeighbors-1 or neighborCount[j] > maxNeighbors-1):
                    print("maxNeighbors update")
                    newMaxNeighbors = np.max([neighborCount[i], neighborCount[j]])
                    neighbors = np.pad(neighbors, (0, newMaxNeighbors-maxNeighbors), 'constant', constant_values=-1)[:numParticles]
                    maxNeighbors = newMaxNeighbors
    return neighbors

def computeDeltas(pos, boxSize):
    numParticles = pos.shape[0]
    deltas = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    deltas += boxSize / 2
    deltas %= boxSize
    deltas -= boxSize / 2
    return deltas

def computeTimeDistances(pos1, pos2, boxSize):
    distances = np.zeros((pos1.shape[0], pos1.shape[0]))
    for i in range(pos.shape[0]):
        for j in range(i):
            delta = pbcDistance(pos1[i], pos2[j], boxSize)
            distances[i,j] = np.linalg.norm(delta)
    return distances

def getPairCorr(pos, boxSize, bins, minRad):
    #distance = np.triu(computeDistances(pos, boxSize),1)
    distance = computeDistances(pos, boxSize)
    distance = distance.flatten()
    distance = distance[distance>0]
    pairCorr, edges = np.histogram(distance, bins=bins)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    return pairCorr / (2 * np.pi * binCenter)

def polarPos(r, alpha):
    return r * np.array([np.cos(alpha), np.sin(alpha)])

def checkAngle(alpha):
    if(alpha < 0):
        alpha += 2*np.pi
    elif(alpha > 2*np.pi):
        alpha -= 2*np.pi
    return alpha

def computeAdjacencyMatrix(dirName, numParticles=None):
    if(numParticles==None):
        numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    adjacency = np.zeros((numParticles, numParticles), dtype=int)
    for i in range(numParticles):
        adjacency[i,contacts[i,np.argwhere(contacts[i]!=-1)[:,0]].astype(int)] = 1
    return adjacency

############################ correlation functions #############################
def computeIsoCorrFunctions(pos1, pos2, boxSize, waveVector, scale, oneDim = False):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    delta = np.linalg.norm(delta, axis=1)
    if(oneDim == True):
        delta = pos1[:,0] - pos2[:,0]
        delta -= np.mean(delta)
    msd = np.mean(delta**2)/scale
    isf = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    chi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isf*isf
    return msd, isf, chi4

def computeCorrFunctions(pos1, pos2, boxSize, waveVector, scale):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    Sq = []
    angleList = np.arange(0, 2*np.pi, np.pi/8)
    for angle in angleList:
        q = np.array([np.cos(angle), np.sin(angle)])
        Sq.append(np.mean(np.exp(1j*waveVector*np.sum(np.multiply(q, delta), axis=1))))
    Sq = np.array(Sq)
    ISF = np.real(np.mean(Sq))
    Chi4 = np.real(np.mean(Sq**2) - np.mean(Sq)**2)
    delta = np.linalg.norm(delta, axis=1)
    MSD = np.mean(delta**2)/scale
    isoISF = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    isoChi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isoISF*isoISF
    alpha2 = np.mean(delta**4)/(2*np.mean(delta**2)**2) - 1
    alpha2new = np.mean(delta**2)/(2*np.mean(1/delta**2)) - 1
    return MSD, ISF, Chi4, isoISF, isoChi4, alpha2, alpha2new

def computeSingleParticleISF(pos1, pos2, boxSize, waveVector, scale):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    delta = np.linalg.norm(delta, axis=1)
    #msd = delta**2
    isf = np.sin(waveVector * delta) / (waveVector * delta)
    return isf

def computeShapeCorrFunction(shape1, shape2):
    return (np.mean(shape1 * shape2) - np.mean(shape1)**2) / (np.mean(shape1**2) - np.mean(shape1)**2)

def computeVelCorrFunction(vel1, vel2):
    return np.mean(np.sum(vel1 * vel2, axis=1))

def computeVelCorrFunctions(pos1, pos2, vel1, vel2, dir1, dir2, waveVector, numParticles):
    speed1 = np.linalg.norm(vel1, axis=1)
    velNorm1 = np.mean(speed1)
    speed2 = np.linalg.norm(vel2, axis=1)
    velNorm2 = np.mean(speed2)
    speedCorr = np.mean(speed1 * speed2)
    velCorr = np.mean(np.sum(np.multiply(vel1, vel2)))
    dirCorr = np.mean(np.sum(np.multiply(dir1, dir2)))
    # compute velocity weighted ISF
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    velSq = []
    angleList = np.arange(0, 2*np.pi, np.pi/8)
    for angle in angleList:
        unitk = np.array([np.cos(angle), np.sin(angle)])
        k = unitk*waveVector
        weight = np.exp(1j*np.sum(np.multiply(k,delta), axis=1))
        s1 = np.sum(vel1[:,0]*vel2[:,0]*weight)
        s2 = np.sum(vel1[:,0]*vel2[:,1]*weight)
        s3 = np.sum(vel1[:,1]*vel2[:,1]*weight)
        vsf = np.array([[s1, s2], [s2, s3]])
        velSq.append(np.dot(np.dot(unitk, vsf), unitk))
    velISF = np.real(np.mean(velSq))/numParticles
    return speedCorr, velCorr, dirCorr, velISF

def computeSusceptibility(pos1, pos2, field, waveVector, scale):
    delta = pos1[:,0] - pos2[:,0]
    delta -= np.mean(delta)
    chi = np.mean(delta / field)
    isf = np.exp(1.j*waveVector*delta/field)
    chiq = np.mean(isf**2) - np.mean(isf)**2
    return chi / scale, np.real(chiq)

def computeLocalAreaGrid(pos, rad, xbin, ybin, localArea):
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localArea[x, y] += np.pi*rad[pId]**2

def computeLocalAreaAndNumberGrid(pos, rad, xbin, ybin, localArea, localNumber):
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localArea[x, y] += np.pi*rad[pId]**2
                        localNumber[x, y] += 1

def computeLocalTempGrid(pos, vel, xbin, ybin, localTemp): #this works only for 2d
    counts = np.zeros((localTemp.shape[0], localTemp.shape[1]))
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localTemp[x, y] += np.linalg.norm(vel[pId])**2
                        counts[x, y] += 1
    localTemp[localTemp>0] /= counts[localTemp>0]*2

def computeTau(data, index=2, threshold=np.exp(-1), normalized=False):
    if(normalized == True):
        data[:,index] /= data[0,index]
    relStep = np.argwhere(data[:,index]>threshold)[-1,0]
    if(relStep + 1 < data.shape[0]):
        t1 = data[relStep,0]
        t2 = data[relStep+1,0]
        ISF1 = data[relStep,index]
        ISF2 = data[relStep+1,index]
        slope = (ISF2 - ISF1)/(t2 - t1)
        intercept = ISF2 - slope * t2
        return (np.exp(-1) - intercept)/slope
    else:
        return data[relStep,0]

def computeDeltaChi(data):
    maxStep = np.argmax(data[:,5])
    maxChi = np.max(data[:,5])
    if(maxStep + 1 < data.shape[0]):
        # find values of chi above the max/2
        domeSteps = np.argwhere(data[:,5]>maxChi*0.5)
        t1 = domeSteps[0]
        t2 = domeSteps[-1]
        return t2 - t1
    else:
        return 0

def getDBClusterLabels(pos, boxSize, eps, min_samples, contacts, contactFilter=False):
    numParticles = pos.shape[0]
    distance = computeDistances(pos, boxSize)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance)
    labels = db.labels_
    if(contactFilter == 'contact'):
        connectLabel = np.zeros(numParticles)
        for i in range(numParticles):
            if(np.sum(contacts[i]!=-1)>1):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(np.sum(contacts[c]!=-1)>1):
                            # this is at least a three particle cluster
                            connectLabel[i] = 1
        labels[connectLabel==0] = -1
    return labels

def getNoClusterLabel(labels, contacts):
    noLabels = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        if(labels[i] != -1 and np.sum(contacts[i]) < 2):
            noLabels[i] = 1
    return noLabels


############################## Fourier Analysis ################################
def getStructureFactor(pos, q, numParticles):
    sfList = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        sf = []
        for i in range(theta.shape[0]):
            k = q[j]*np.array([np.cos(theta[i]), np.sin(theta[i])])
            posDotK = np.dot(pos,k)
            sf.append(np.sum(np.exp(-1j*posDotK))*np.sum(np.exp(1j*posDotK)))
        sfList[j] = np.real(np.mean(sf))/numParticles
    return sfList

def getVelocityStructureFactor(pos, vel, q, numParticles):
    velsfList = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        velsf = []
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            s1 = np.sum(vel[:,0]*vel[:,0]*np.exp(-1j*posDotK))*np.sum(vel[:,0]*vel[:,0]*np.exp(1j*posDotK))
            s2 = np.sum(vel[:,0]*vel[:,1]*np.exp(-1j*posDotK))*np.sum(vel[:,0]*vel[:,1]*np.exp(1j*posDotK))
            s3 = np.sum(vel[:,1]*vel[:,1]*np.exp(-1j*posDotK))*np.sum(vel[:,1]*vel[:,1]*np.exp(1j*posDotK))
            vsf = np.array([[s1, s2], [s2, s3]])
            velsf.append(np.dot(np.dot(unitk, vsf), unitk))
        velsfList[j] = np.real(np.mean(velsf))/numParticles
    return velsfList

def getSpaceFourierEnergy(pos, vel, epot, q, numParticles):
    kq = np.zeros(q.shape[0])
    uq = np.zeros(q.shape[0])
    kcorr = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        ktemp = []
        utemp = []
        kctemp = []
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            ekin = 0.5*np.linalg.norm(vel, axis=1)**2
            ktemp.append(np.sum(ekin.T*np.exp(-1j*posDotK)))
            utemp.append(np.sum(epot.T*np.exp(1j*posDotK))*np.sum(epot.T*np.exp(-1j*posDotK)))
            # correlations
            kctemp.append(np.sum(ekin.T*np.exp(1j*posDotK))*np.sum(ekin.T*np.exp(-1j*posDotK)))
        kq[j] = np.abs(np.mean(ktemp))/numParticles
        uq[j] = np.abs(np.mean(utemp))/numParticles
        kcorr[j] = np.abs(np.mean(kctemp))/numParticles
    return kq, uq, kcorr

def getTimeFourierEnergy(dirName, dirList, dirSpacing, numParticles):
    timeStep = readFromParams(dirName, "dt")
    numSteps = dirList.shape[0]
    freq = fftfreq(numSteps, dirSpacing*timeStep)
    energy = np.zeros((numSteps,numParticles,2))
    corre = np.zeros((numSteps,numParticles,2))
    initialEpot = np.array(np.loadtxt(dirName + os.sep + "t0/particleEnergy.dat"), dtype=np.float64)
    initialVel = np.array(np.loadtxt(dirName + os.sep + "t0/particleVel.dat"), dtype=np.float64)
    initialEkin = 0.5*np.linalg.norm(initialVel, axis=1)**2
    # collect instantaneous energy for 10 particles
    for i in range(numSteps):
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
        epot = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleEnergy.dat"), dtype=np.float64)
        ekin = 0.5*np.linalg.norm(vel, axis=1)**2
        energy[i,:,0] = ekin
        energy[i,:,1] = epot
        corre[i,:,0] = ekin*initialEkin
        corre[i,:,1] = epot*initialEpot
    # compute fourier transform and average over particles
    energyf = np.zeros((numSteps, 3), dtype=complex)
    corref = np.zeros((numSteps, 3), dtype=complex)
    for pId in range(numParticles):
        energyf[:,0] += fft(energy[:,pId,0])
        energyf[:,1] += fft(energy[:,pId,1])
        energyf[:,2] += fft(energy[:,pId,0] + energy[:,pId,1])
        # correlations
        corref[:,0] += fft(corre[:,pId,0])
        corref[:,1] += fft(corre[:,pId,1])
        corref[:,2] += fft(corre[:,pId,0] + corre[:,pId,1])
    energyf /= numParticles
    energyf = energyf[np.argsort(freq)]
    corref /= numParticles
    corref = corref[np.argsort(freq)]
    freq = np.sort(freq)
    return np.column_stack((freq, np.abs(energyf)*2/numSteps, np.abs(corref)*2/numSteps))

def getSpaceFourierVelocity(pos, vel, q, numParticles):
    vq = np.zeros((q.shape[0],2))
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        vqtemp = np.zeros(2)
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            vqx = np.sum(vel[:,0].T*np.exp(-1j*posDotK))
            vqy = np.sum(vel[:,1].T*np.exp(-1j*posDotK))
            vqxy = np.array([vqx, vqy])
            vqtemp[0] += np.mean(np.abs(np.dot(vqxy, unitk))**2)
            vqtemp[1] += np.mean(np.abs(np.cross(vqxy, unitk))**2)
        vq[j] = vqtemp/theta.shape[0]
    return vq

def getTimeFourierVel(dirName, dirList, dirSpacing, numParticles):
    timeStep = readFromParams(dirName, "dt")
    numSteps = dirList.shape[0]
    freq = fftfreq(numSteps, dirSpacing*timeStep)
    veltot = np.zeros((numSteps,numParticles,2))
    # collect instantaneous energy for 10 particles
    for i in range(numSteps):
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
        veltot[i] = vel
    # compute fourier transform and average over particles
    velf = np.zeros((numSteps,2), dtype=complex)
    for pId in range(numParticles):
        velf[:,0] += fft(veltot[:,pId,0])
        velf[:,1] += fft(veltot[:,pId,1])
    velf /= numParticles
    velf = velf[np.argsort(freq)]
    velfSquared1 = np.mean(np.abs(velf)**2,axis=1)*2/numSteps
    # compute fourier transform and average over particles
    velf = np.zeros((numSteps,2))
    for pId in range(numParticles):
        velf[:,0] += np.abs(fft(veltot[:,pId,0]))**2
        velf[:,1] += np.abs(fft(veltot[:,pId,1]))**2
    velf /= numParticles
    velf = velf[np.argsort(freq)]
    velSquared2 = np.mean(velf,axis=1)*2/numSteps
    freq = np.sort(freq)
    return np.column_stack((freq, velfSquared1, velSquared2))


############################### read from files ################################
def getDirectories(dirName):
    listDir = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "bab" and dir != "dynamics")):
            listDir.append(dir)
    return listDir

def getOrderedDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "bab" and dir != "dynamics")):
            listDir.append(dir)
            listScalar.append(dir.strip('t'))
    listScalar = np.array(listScalar, dtype=np.int64)
    listDir = np.array(listDir)
    listDir = listDir[np.argsort(listScalar)]
    listScalar = np.sort(listScalar)
    return listDir, listScalar

def getDirSep(dirName, fileName):
    if(os.path.exists(dirName + os.sep + fileName + ".dat")):
        return "/"
    else:
        return "/../"

def readFromParams(dirName, paramName):
    with open(dirName + os.sep + "params.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                return float(scalarString)

def readFromDynParams(dirName, paramName):
    with open(dirName + os.sep + "dynParams.dat") as file:
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

def readVelPair(dirName, index1, index2):
    pVel1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleVel.dat"))
    pVel2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleVel.dat"))
    return pVel1, pVel2

def readPair(dirName, index1, index2):
    pPos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particlePos.dat"))
    pos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "positions.dat"))
    pPos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particlePos.dat"))
    pos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "positions.dat"))
    return pPos1, pos1, pPos2, pos2

def readDirectorPair(dirName, index1, index2):
    pAngle1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleAngles.dat"))
    pAngle2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleAngles.dat"))
    pDir1 = np.array([np.cos(pAngle1), np.sin(pAngle1)]).T
    pDir2 = np.array([np.cos(pAngle2), np.sin(pAngle2)]).T
    return pDir1, pDir2

def computeParticleVelocities(vel, nv):
    numParticles = nv.shape[0]
    pVel = np.zeros((numParticles,2))
    firstVertex = 0
    for pId in range(numParticles):
        pVel[pId] = [np.mean(vel[firstVertex:firstVertex+nv[pId],0]), np.mean(vel[firstVertex:firstVertex+nv[pId],1])]
        firstVertex += nv[pId]
    return pVel

def getPBCPositions(fileName, boxSize):
    pos = np.array(np.loadtxt(fileName), dtype=np.float64)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    return pos

if __name__ == '__main__':
    print("library for correlation function utilities")


# the formula to compute the drift-subtracted msd is
#delta = np.linalg.norm(pos1 - pos2, axis=1)
#drift = np.linalg.norm(np.mean(pos1 - pos2, axis=0)**2)
#msd = np.mean(delta**2) - drift
# equivalent to
#drift = np.mean(delta)**2
# in one dimension
#gamma2 = (1/3) * np.mean(delta**2) * np.mean(1/delta**2) - 1


#distances = np.zeros((pos.shape[0], pos.shape[0]))
#for i in range(pos.shape[0]):
#    for j in range(i):
#        delta = pbcDistance(pos[i], pos[j], boxSize)
#        distances[i,j] = np.linalg.norm(delta)
