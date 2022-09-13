'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
import os

############################## general utilities ###############################
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

# the formula to compute the drift-subtracted msd is
#delta = np.linalg.norm(pos1 - pos2, axis=1)
#drift = np.linalg.norm(np.mean(pos1 - pos2, axis=0)**2)
#msd = np.mean(delta**2) - drift
# equivalent to
#drift = np.mean(delta)**2
# in one dimension
#gamma2 = (1/3) * np.mean(delta**2) * np.mean(1/delta**2) - 1
def computeCorrFunctions(pos1, pos2, boxSize, waveVector, scale, oneDim = False):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    delta = np.linalg.norm(delta, axis=1)
    if(oneDim == True):
        delta = pos1[:,0] - pos2[:,0]
        delta -= np.mean(delta)
    msd = np.mean(delta**2)
    isf = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    chi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isf*isf
    return msd / scale, isf, chi4

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

def computeSusceptibility(pos1, pos2, field, waveVector, scale):
    delta = pos1[:,0] - pos2[:,0]
    delta -= np.mean(delta)
    chi = np.mean(delta / field)
    isf = np.exp(1.j*waveVector*delta/field)
    chiq = np.mean(isf**2) - np.mean(isf)**2
    return chi / scale, np.real(chiq)

def computeLocalAreaGrid(pos, area, xbin, ybin, localArea):
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localArea[x, y] += area[pId]
                        

############################### read from files ################################
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
    pVel1 = np.array([np.cos(pAngle1), np.sin(pAngle1)]).T
    pVel2 = np.array([np.cos(pAngle2), np.sin(pAngle2)]).T
    return pVel1, pVel2

def computeParticleVelocities(vel, nv):
    numParticles = nv.shape[0]
    pVel = np.zeros((numParticles,2))
    firstVertex = 0
    for pId in range(numParticles):
        pVel[pId] = [np.mean(vel[firstVertex:firstVertex+nv[pId],0]), np.mean(vel[firstVertex:firstVertex+nv[pId],1])]
        firstVertex += nv[pId]
    return pVel

if __name__ == '__main__':
    print("library for correlation function utilities")
