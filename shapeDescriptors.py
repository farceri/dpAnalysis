'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import sys
import os

def pbcDistance(r1, r2, boxSize):
    delta = r1 - r2
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

def computeCorrelation(data1, data2):
    return (data2 - data1)**2

def computePDF(data, bins, density=True):
    pdf, edges = np.histogram(data, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pdf, centers

def plotDistributions(data, numBins=20, xlabel="$e_x$", ylabel="$e_y$", figureName="pdf"):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), sharey = True, dpi = 120)
    for m in range(data.shape[1]):
        pdf, bins = computePDF(data[:,m], np.linspace(np.min(data[:,m]), np.max(data[:,m]), numBins))
        ax[m].plot(bins, pdf, linewidth=1.2, color = 'k')
        ax[m].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel(xlabel, fontsize=17)
    ax[1].set_xlabel(ylabel, fontsize=17)
    ax[0].set_ylabel("$PDF$", fontsize=17)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    plt.savefig("/home/francesco/Pictures/dpm/" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def getAreaAndPerimeterList(pos, boxSize, nv):
    numParticles = nv.shape[0]
    perimeter = np.zeros(numParticles)
    area = np.zeros(numParticles)
    firstVertex = 0
    for pId in range(numParticles):
        currentPos = pos[firstVertex,:]
        idList = np.arange(firstVertex, firstVertex+nv[pId], 1)
        for vId in idList:
            nextId = vId + 1
            if(vId == idList[-1]):
                nextId = idList[0]
            delta = pbcDistance(pos[nextId,:], currentPos, boxSize)
            nextPos = pos[nextId,:] + delta
            area[pId] += currentPos[0] * nextPos[1] - nextPos[0] * currentPos[1]
            perimeter[pId] += np.linalg.norm(delta)
            currentPos = nextPos
        firstVertex += nv[pId]
    area = np.abs(area) * 0.5
    return area, perimeter

def getShapeDirections(dirName, boxSize, nv, eigstimeseigv=False):
    numParticles = nv.shape[0]
    # find largest eigenvector of moment of inertia
    eigs, eigv, pPos = computeInertiaTensor(dirName, boxSize, nv, plot=False)
    pPos -= np.floor(pPos/boxSize) * boxSize
    eigvmax = np.zeros((numParticles, 2))
    for i in range(numParticles):
        eigvmax[i] = eigv[i,np.argmax(eigs[i])]
        if(eigstimeseigv == True):
            eigvmax[i] *= np.max(eigs[i])
    return eigvmax, pPos

def getOrientationCosangleList(dirName, boxSize, nv):
    numParticles = nv.shape[0]
    eigvmax, pPos = getShapeDirections(dirName, boxSize, nv)
    x = np.array([1,0])
    cosangle = np.zeros(numParticles)
    for pId in range(numParticles):
        cosangle[pId] = np.abs(np.dot(eigvmax[pId], x))
    return cosangle

def computeShapeMoments(dirName, boxSize, nv, numBins = 20):
    numParticles = nv.shape[0]
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    area, perimeter = getAreaAndPerimeterList(pos, boxSize, nv)
    shapeParam = perimeter**2/(4*np.pi*area)
    moment = np.zeros((numParticles,3))
    moment[:,0] = np.mean(shapeParam)
    moment[:,1] = np.mean(shapeParam**2)
    moment[:,2] = np.mean(shapeParam**3)
    data1 = moment[:,0]
    data2 = np.sqrt(moment[:,1] - moment[:,0]**2)
    data3 = (moment[:,2] - moment[:,0]**3)**(1/3)
    return data1, data2, data3

def computeStressTensor(dirName, nv, numBins = 20, plot=True):
    numParticles = nv.shape[0]
    numVertices = np.sum(nv)
    stress = np.array(np.loadtxt(dirName + os.sep + "particleStress.dat"))
    eigs = np.zeros((numParticles,2))
    eigv = np.zeros((numParticles,2,2))
    firstVertex = 0
    for pId in range(numParticles):
        eigs[pId], eigv[pId] = np.linalg.eigh(stress[pId].reshape((2,2)))
        eigs[pId] /= np.sqrt(eigs[pId,0]*eigs[pId,1])
    if(plot==True):
        plotDistributions(eigs, numBins=20, xlabel="$e_x$", ylabel="$e_y$", figureName="stressTensor-pdf")
    return eigs, eigv

def computeInertiaTensor(dirName, boxSize, nv, numBins = 20, plot=True):
    numParticles = nv.shape[0]
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    moment = np.zeros((numParticles,4))
    eigs = np.zeros((numParticles,2))
    eigv = np.zeros((numParticles,2,2))
    firstVertex = 0
    for pId in range(numParticles):
        idList = np.arange(firstVertex, firstVertex+nv[pId], 1)
        for vId in idList:
            delta = pbcDistance(pos[vId,:], pPos[pId], boxSize)
            moment[pId,0] += delta[0]**2
            moment[pId,1] += delta[0]*delta[1]
            moment[pId,2] += delta[1]*delta[0]
            moment[pId,3] += delta[1]**2
        moment[pId] /= nv[pId]
        firstVertex += nv[pId]
        eigs[pId], eigv[pId] = np.linalg.eigh(moment[pId].reshape((2,2)))
        eigs[pId] /= np.sqrt(eigs[pId,0]*eigs[pId,1])
    if(plot==True):
        plotDistributions(eigs, numBins=20, xlabel="$e_x$", ylabel="$e_y$", figureName="inertia-pdf")
    return eigs, eigv, pPos

def computeStretchTensor(dirName, boxSize, nv, numBins = 20, plot=True):
    numParticles = nv.shape[0]
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    moment = np.zeros((numParticles,4))
    eigs = np.zeros((numParticles,2))
    firstVertex = 0
    for pId in range(numParticles):
        currentPos = pos[firstVertex,:]
        idList = np.arange(firstVertex, firstVertex+nv[pId], 1)
        for vId in idList:
            nextId = vId + 1
            if(vId == idList[-1]):
                nextId = idList[0]
            delta = pbcDistance(pos[nextId,:], currentPos, boxSize)
            nextPos = pos[nextId,:] + delta
            moment[pId,0] += delta[0]**2
            moment[pId,1] += delta[0]*delta[1]
            moment[pId,2] += delta[1]*delta[0]
            moment[pId,3] += delta[1]**2
            currentPos = nextPos
        moment[pId] /= nv[pId]
        firstVertex += nv[pId]
        eigs[pId], _ = np.linalg.eig(moment[pId].reshape((2,2)))
    if(plot==True):
        plotDistributions(eigs, numBins=20, xlabel="$e_x$", ylabel="$e_y$", figureName="stretch-pdf")
    return eigs

def computeParticleElongation(pos, boxSize):
    distances = computeDistances(pos, boxSize)
    longestDistance = 0
    for i in range(pos.shape[0]):
        for j in range(i):
            if(distances[i,j] > longestDistance):
                longestDistance = distances[i,j]
                longestDelta = pbcDistance(pos[i], pos[j], boxSize)
    longestPerpendicularDistance = 0
    highestSinangle = 0
    for i in range(pos.shape[0]):
        for j in range(i):
            delta = pbcDistance(pos[i], pos[j], boxSize)
            sinangle = np.sin(np.dot(delta, longestDelta) / (distances[i,j] * longestDistance))
            if(sinangle > highestSinangle):
                highestSisngle = sinangle
                longestPerpendicularDistance = distances[i,j]
    return longestDistance / longestPerpendicularDistance

def computeElongation(dirName, boxSize, nv):
    numParticles = nv.shape[0]
    elongation = []
    pos = np.array(np.loadtxt(dirName + os.sep + "/positions.dat"))
    firstVertex = 0
    for pId in range(numParticles):
        currentPos = pos[firstVertex:firstVertex+nv[pId],:]
        elongation.append(computeParticleElongation(currentPos, boxSize))
        firstVertex += nv[pId]
    return elongation / numParticles

def getVectorFieldAlignement(dirName, field, angleTh=15, alignedTh=2, numParticles=128):
    # filter and build a list of intensity based on contact alignment
    contacts = np.array(np.loadtxt(dirName + os.sep + "contacts.dat"), dtype=int)
    intensity = np.zeros(numParticles)
    dotProductTh = np.cos(np.radians(angleTh))
    for i in range(numParticles):
        numContacts = np.sum(contacts[i] != -1)
        numAligned = 0
        aligned = []
        for c in range(contacts[i].shape[0]):
            if(contacts[i,c] != -1):
                dotProduct = np.abs(np.dot(field[i], field[contacts[i,c]]))
                if(dotProduct > dotProductTh):#10 degrees
                    numAligned += 1
                    aligned.append(contacts[i,c])
                    if(numAligned == alignedTh):
                        intensity[i] = 1
                        intensity[aligned] = 1
                        c = contacts[i].shape[0]
    return intensity

def clusterVectorField(dirName, field, intensity, angleTh=15, numParticles=128):
    contacts = np.array(np.loadtxt(dirName + os.sep + "contacts.dat"), dtype=int)
    #contacts = np.array(np.loadtxt(dirName + os.sep + "neighbors.dat"), dtype=int)
    dotProductTh = np.cos(np.radians(angleTh))
    alignedList = np.argwhere(intensity==1).flatten()
    if(alignedList.shape[0]==0):
        return [], []
    else:
        particleLabel = -np.ones(numParticles, dtype=int)
        clusterlabel = 0
        clusterList = []
        # let's define the first particle to belong to the first cluster
        particleLabel[alignedList[0]] = clusterlabel
        clusterList.append(clusterlabel)
        for i in range(alignedList.shape[0]):
            for j in range(alignedList.shape[0]):
                # get all the particles aligned to particle i
                dotProduct = np.abs(np.dot(field[alignedList[i]], field[alignedList[j]]))
                if(dotProduct > dotProductTh): # if particles are aligned
                    for c in range(contacts[alignedList[i]].shape[0]):
                        if(contacts[alignedList[i],c] == alignedList[j]): # if particles are in contact
                            # the two particles belong to the same cluster
                            foundExistingCluster = False
                            for label in clusterList:
                                particleInCluster = np.argwhere(particleLabel==label).flatten()
                                for pId in particleInCluster:
                                    dotProduct = np.abs(np.dot(field[alignedList[i]], field[pId]))
                                    if(dotProduct > dotProductTh): # if particles are aligned to an existing cluster
                                        for s in range(contacts[pId].shape[0]):
                                            if(contacts[pId,s] == alignedList[i] or contacts[pId,s] == alignedList[j]):
                                                particleLabel[alignedList[i]] = label
                                                particleLabel[alignedList[j]] = label
                                                foundExistingCluster = True
                            if(foundExistingCluster == False):
                                clusterlabel += 1
                                particleLabel[alignedList[i]] = clusterlabel
                                particleLabel[alignedList[j]] = clusterlabel
                                clusterList.append(clusterlabel)
        # compute cluster average shape director
        clusterList = np.array(clusterList)
        particleLabel = np.array(particleLabel)
        clusterAngle = np.zeros(clusterList.shape[0])
        x = np.array([1,0])
        for label in clusterList:
            if(particleLabel[particleLabel==label].shape[0]>3):
                clusterAngle[label] = np.degrees(np.arccos(np.abs(np.dot(field[np.argwhere(particleLabel==label)[0,0]], x))))
        return clusterList, particleLabel, clusterAngle

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "moments"):
        numBins = int(sys.argv[3])
        computeShapeMoments(dirName, numBins)

    elif(whichCorr == "inertia"):
        numBins = int(sys.argv[3])
        computeInertiaTensor(dirName, numBins)

    elif(whichCorr == "stretch"):
        numBins = int(sys.argv[3])
        computeStretchTensor(dirName, numBins)

    elif(whichCorr == "elong"):
            computeElongation(dirName)

    else:
        print("Please specify the type of correlation you want to compute")
