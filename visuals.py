'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from sklearn.cluster import KMeans
import itertools
import sys
import os
import shapeDescriptors
import shapeGraphics
import spCorrelation as spCorr
import utilsCorr as ucorr

def setAxes2D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setPackingAxes(boxSize, ax):
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setGridAxes(bins, ax):
    xBounds = np.array([bins[0], bins[-1]])
    yBounds = np.array([bins[0], bins[-1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setBigBoxAxes(boxSize, ax, delta=0.1):
    xBounds = np.array([-delta, boxSize[0]+delta])
    yBounds = np.array([-delta, boxSize[1]+delta])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def getRadColorList(rad):
    colorList = cm.get_cmap('viridis', rad.shape[0])
    colorId = np.zeros((rad.shape[0], 4))
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    return colorId

def getEkinColorList(vel):
    colorList = cm.get_cmap('Greys', vel.shape[0])
    colorId = np.zeros((vel.shape[0], 4))
    count = 0
    ekin = np.linalg.norm(vel,axis=1)**2
    print("kinetic energy: ", np.sum(ekin)/(vel.shape[0]*2))
    for particleId in np.argsort(ekin):
        colorId[particleId] = colorList(count/ekin.shape[0])
        count += 1
    return colorId

def getClusterColorList(pos, nClusters=10):
    y_pred = KMeans(n_clusters=nClusters).fit_predict(pos)
    colorList = cm.get_cmap('tab20', nClusters)
    colorId = np.zeros((pos.shape[0], 4))
    for particleId in range(pos.shape[0]):
        colorId[particleId] = colorList(y_pred[particleId])
    return colorId

def plotSPPacking(dirName, figureName, ekmap=False, quiver=False, cluster=False, nClusters=10, alpha = 0.6):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    if(os.path.exists(dirName + os.sep + "particleRad.dat")):
        sep = "/"
    else:
        sep = "../"
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setBigBoxAxes(boxSize, ax, 0.05)
    if(cluster==True):
        colorId = getClusterColorList(pos, nClusters)
    elif(ekmap==True):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        colorId = getEkinColorList(vel)
    else:
        colorId = getRadColorList(rad)
    if(quiver==True):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        #vel *= 5
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(quiver==True):
            ax.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth = 0.7))
            vx = vel[particleId,0]
            vy = vel[particleId,1]
            ax.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=20)#width=0.002, scale=3)
        else:
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth='0.5'))
        #plt.pause(1)
    if(cluster==True):
        figureName = "/home/francesco/Pictures/soft/packings/cluster" + figureName + ".png"
    elif(ekmap==True):
        figureName = "/home/francesco/Pictures/soft/packings/ekmap-" + figureName + ".png"
    elif(quiver==True):
        figureName = "/home/francesco/Pictures/soft/packings/velmap-" + figureName + ".png"
    else:
        figureName = "/home/francesco/Pictures/soft/packings/" + figureName + ".png"
    plt.savefig(figureName, transparent=False, format = "png")
    plt.show()

def plotSoftParticles(ax, pos, rad, alpha = 0.6, colorMap = True, lw = 0.5):
    colorId = np.zeros((rad.shape[0], 4))
    if(colorMap == True):
        colorList = cm.get_cmap('viridis', rad.shape[0])
    else:
        colorList = cm.get_cmap('Reds', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth = lw))

def plotSoftParticlesSubSet(ax, pos, rad, firstIndex, alpha = 0.6, colorMap = True, lw = 0.5):
    colorId = np.zeros((rad.shape[0], 4))
    if(colorMap == True):
        colorList = cm.get_cmap('viridis', rad.shape[0])
    else:
        colorList = cm.get_cmap('Reds', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    colorId[:firstIndex] = [0,0,0,1]
    alphaId = np.ones(colorId.shape[0])
    alphaId[firstIndex:] = alpha
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alphaId[particleId], linewidth = lw))

def plotSoftParticleQuiverVel(axFrame, pos, vel, rad, alpha = 0.6, maxVelList = []):#122, 984, 107, 729, 59, 288, 373, 286, 543, 187, 6, 534, 104, 347]):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        vx = vel[particleId,0]
        vy = vel[particleId,1]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth = 0.7))
        axFrame.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=10)#width=0.003, scale=1, headwidth=5)
        #for j in range(13):
        #    if(particleId == maxVelList[j]):
        #        axFrame.quiver(x, y, vx, vy, facecolor='k', width=0.003, scale=1, headwidth=5)

def plotSoftParticleCluster(axFrame, pos, rad, clusterList, alpha = 0.6):
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(clusterList[particleId] == 1):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth = 0.7))
        else:
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[1,1,1], alpha=alpha, linewidth = 0.7))

def makeSoftParticleFrame(dirName, rad, boxSize, figFrame, frames, subSet = False, firstIndex = 0, npt = False, quiver = False, cluster = False):
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    if(subSet == "subset"):
        plotSoftParticlesSubSet(axFrame, pos, rad, firstIndex)
    elif(quiver == "quiver"):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        plotSoftParticleQuiverVel(axFrame, pos, vel, rad)
    elif(cluster == "cluster"):
        #if(os.path.exists(dirName + os.sep + "clusterList.dat")):
        #    clusterList = np.loadtxt(dirName + os.sep + "clusterList.dat")[:,1]
            #clusterList = np.loadtxt(dirName + os.sep + "deepList.dat")
        #else:
        clusterList = spCorr.searchClusters(dirName, numParticles=rad.shape[0], cluster=cluster)
        plotSoftParticleCluster(axFrame, pos, rad, clusterList)
    else:
        if(npt == "npt"):
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
        plotSoftParticles(axFrame, pos, rad)
    plt.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPPackingVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04, logSpaced = False, subSet = False, firstIndex = 0, npt = False, quiver = False, cluster = False):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    if(logSpaced == False):
        stepList = shapeGraphics.getStepList(numFrames, firstStep, stepFreq)
    else:
        stepList = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir) and dir != "dynamics"):
                stepList.append(int(dir[1:]))
        stepList = np.array(np.sort(stepList))
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # the first configuration gets two frames for better visualization
    makeSoftParticleFrame(dirName + os.sep + "t" + str(stepList[0]), rad, boxSize, figFrame, frames, subSet, firstIndex, npt, quiver, cluster)
    vel = []
    for i in stepList:
        dirSample = dirName + os.sep + "t" + str(i)
        makeSoftParticleFrame(dirSample, rad, boxSize, figFrame, frames, subSet, firstIndex, npt, quiver, cluster)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    if(quiver=="quiver"):
        figureName = "velmap-" + figureName
    anim.save("/home/francesco/Pictures/soft/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeVelFieldFrame(dirName, numBins, bins, boxSize, numParticles, figFrame, frames):
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setGridAxes(bins, axFrame)
    grid, field = spCorr.computeVelocityField(dirName, numBins, plot=False, boxSize=boxSize, numParticles=numParticles)
    axFrame.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.002, scale=3)
    plt.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPVelFieldVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04, numBins=20):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    _, stepList = ucorr.getOrderedDirectories(dirName)
    #timeList = timeList.astype(int)
    stepList = stepList[np.argwhere(stepList%stepFreq==0)[:,0]]
    stepList = stepList[:numFrames]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    setGridAxes(bins, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    # the first configuration gets two frames for better visualization
    makeVelFieldFrame(dirName + os.sep + "t0", numBins, bins, boxSize, numParticles, figFrame, frames)
    vel = []
    for i in stepList:
        dirSample = dirName + os.sep + "t" + str(i)
        makeVelFieldFrame(dirSample, numBins, bins, boxSize, numParticles, figFrame, frames)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/soft/packings/velfield-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotDeformableParticles(ax, pos, rad, nv, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, ls = '-', lw = 0.5):
    start = 0
    colorList = cm.get_cmap('viridis', nv.shape[0])
    colorId = np.zeros((nv.shape[0], 4))
    count = 0
    for particleId in np.argsort(nv):
        colorId[particleId] = colorList(count/nv.shape[0])
        count += 1
    for particleId in range(nv.shape[0]):
        for vertexId in range(nv[particleId]):
            x = pos[start + vertexId,0]
            y = pos[start + vertexId,1]
            r = rad[start + vertexId]
            if(colorMap == True):
                if(faceColor == [0,0.5,1]):
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = colorId[particleId], alpha = alpha, linestyle = ls, linewidth = lw))
                elif(edgeColorMap == True):
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor[particleId], facecolor = faceColor[particleId], alpha = alpha, linestyle = ls, linewidth = lw))
                else:
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = faceColor(particleId/nv.shape[0]), alpha = alpha, linestyle = ls, linewidth = lw))
            else:
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = faceColor, alpha = alpha, linestyle = ls, linewidth = lw))
        start += nv[particleId]

def trackDeformableParticles(ax, pos, rad, nv, edgeColor = [0.3,0.3,0.3], alpha = 0.5, ls = '-', lw = 0.5, trackList = [], highlightList = []):
    start = 0
    trackList = np.array(trackList)
    highlightList = np.array(highlightList)
    for particleId in range(nv.shape[0]):
        for vertexId in range(nv[particleId]):
            x = pos[start + vertexId,0]
            y = pos[start + vertexId,1]
            r = rad[start + vertexId]
            if(np.isin(particleId, trackList)):
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = [0.2,1,0.2], alpha = 1, linestyle = ls, linewidth = lw))
            elif(np.isin(particleId, highlightList)):
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = 'k', alpha = 1, linestyle = ls, linewidth = lw))
            else:
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = [0.9,0.9,0.9], alpha = alpha, linestyle = ls, linewidth = lw))
        start += nv[particleId]

def plotDPMPacking(dirName, figureName, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, save = True, plot = True):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    if(os.path.exists(dirName + os.sep + "radii.dat")):
        sep = "/"
    else:
        sep = "../"
    rad = np.array(np.loadtxt(dirName + sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotDeformableParticles(ax, pos, rad, nv, faceColor, edgeColor, colorMap, edgeColorMap, alpha)
    if(save == True):
        plt.savefig("/home/francesco/Pictures/dpm/packings/" + figureName + ".png", transparent=True, format = "png")
    if(plot == True):
        plt.show()
    else:
        return ax

def compareDPMPackings(dirName1, dirName2, figureName):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    boxSize = np.loadtxt(dirName1 + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax) # packings need to have the same boxSize
    dirNameList = np.array([dirName1, dirName2])
    colorList = ['r', 'b']
    for i in range(dirNameList.shape[0]):
        pos = np.array(np.loadtxt(dirNameList[i] + os.sep + "positions.dat"))
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        rad = np.array(np.loadtxt(dirNameList[i] + os.sep + "radii.dat"))
        nv = np.array(np.loadtxt(dirNameList[i] + os.sep + "numVertexInParticleList.dat"), dtype=int)
        plotDeformableParticles(ax, pos, rad, nv, colorList[i])
    plt.savefig("/home/francesco/Pictures/dpm/packings/" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames):
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    plotDeformableParticles(axFrame, pos, rad, nv, colorMap = True)
    plt.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeDPMPackingVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, logSpaced = False):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    if(logSpaced == False):
        _, stepList = ucorr.getOrderedDirectories(dirName)
        stepList = stepList[np.argwhere(stepList%stepFreq==0)[:,0]]
        stepList = stepList[:numFrames]
    else:
        stepList = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir) and dir != "dynamics"):
                stepList.append(int(dir[1:]))
        stepList = np.array(np.sort(stepList))
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    # the first configuration gets two frames for better visualization
    pos = np.array(np.loadtxt(dirName + os.sep + "t0/positions.dat"))
    makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames)
    numVertices = rad.shape[0]
    for i in stepList:
        pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat"))
        makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def compareDPMPackingsVideo(dirName, fileName, figureName):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    maxStep = 1001
    numFrames = int((maxStep -1)/ 10)
    frameTime = 200
    frames = []
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    rad = np.loadtxt(dirName + os.sep + "radii.dat")
    rad = np.array(rad)
    numVertices = rad.shape[0]
    for i in range(1, maxStep, 10):
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        # second packing
        pos = np.loadtxt(fileName, skiprows=(numVertices*i), max_rows = numVertices, usecols = (0,1))
        pos = np.array(pos)
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        plotDeformableParticles(axFrame, pos, rad, nv, faceColor = [1,1,1], alpha = 1, ls = '--', lw = 1.2)
        # first packing
        pos = np.loadtxt(dirName + os.sep + "pos_step" + str(i) + ".dat")
        pos = np.array(pos)
        plotDeformableParticles(axFrame, pos, rad, nv)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotDPMcolorHOP(dirName, figureName, colorMap = True, alpha = 0.5, save = False):
    psi6 = spCorrelation.computeHexaticOrder(dirName)
    colorList = [[1-np.abs(x), 1-np.abs(x), 1] for x in psi6]
    plotDPMPacking(dirName, figureName, colorList, colorMap = colorMap, alpha = alpha, save = save)

def makeCompressionVideo(dirName, figureName, numFrames = 50):
    phiList = np.sort(os.listdir(dirName))[60::4]
    phiList = phiList[-numFrames:]
    print(phiList)
    numFrames -= 1
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 200
    frames = []
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + phiList[0] + "/boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + phiList[0] + "/numVertexInParticleList.dat", dtype=int))
    numVertices = np.sum(nv)
    colorList = cm.get_cmap('viridis', nv.shape[0])
    for phi in phiList:
        pos = np.array(np.loadtxt(dirName + os.sep + phi + "/positions.dat"))
        rad = np.array(np.loadtxt(dirName + os.sep + phi +"/radii.dat"))
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        plotDeformableParticles(axFrame, pos, rad, nv, faceColor = colorList, colorMap = True)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/comp-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeRearrengementsVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 200
    frames = []
    stepList = shapeGraphics.getStepList(numFrames, firstStep, stepFreq)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    rad = np.loadtxt(dirName + os.sep + "radii.dat")
    rad = np.array(rad)
    contactdiff = shapeGraphics.getContactDiff(dirName, numParticles, stepList)
    rearrangeList = np.argwhere(contactdiff==2)[1][0]
    initialContacts = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/contacts.dat", dtype=int)+1
    initialContacts = np.flip(np.sort(initialContacts, axis=1), axis=1)
    highlightList = (initialContacts[rearrangeList, initialContacts[rearrangeList]>0]-1).tolist()
    rearrangeList = [rearrangeList]
    print(rearrangeList, highlightList)
    pos = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/positions.dat")
    print(pos[rearrangeList], boxSize)
    for i in stepList:
        pos = np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat")
        pos = np.array(pos)
        #pos[:,0] += 0.2*boxSize[0]
        pos[:,1] -= 0.2*boxSize[1]
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        trackDeformableParticles(axFrame, pos, rad, nv, trackList = rearrangeList, highlightList = highlightList)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/rearrange-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotSPDPMPacking(dirName, figureName, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, save = True):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    area = np.array(np.loadtxt(dirName + os.sep + "restAreas.dat"))
    setPackingAxes(boxSize, ax)
    #pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotDeformableParticles(ax, pos, rad, nv, faceColor, edgeColor, colorMap, edgeColorMap, alpha)
    rad = np.array(np.loadtxt(dirName + os.sep + "softRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "softPos.dat"))
    #pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotSoftParticles(ax, pos, rad, alpha = 0.8, colorMap = False, lw = 0.5)
    plt.tight_layout()
    if(save == True):
        plt.savefig("/home/francesco/Pictures/spdpm/" + figureName + ".png", transparent=False, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]
    figureName = sys.argv[3]

    if(whichPlot == "ss"):
        plotSPPacking(dirName, figureName)

    elif(whichPlot == "ssvel"):
        plotSPPacking(dirName, figureName, quiver=True)

    elif(whichPlot == "sscluster"):
        nClusters = int(sys.argv[4])
        plotSPPacking(dirName, figureName, cluster=True, nClusters=nClusters)

    elif(whichPlot == "ssvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "velfield"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        numBins = int(sys.argv[7])
        makeSPVelFieldVideo(dirName, figureName, numFrames, firstStep, stepFreq, numBins=numBins)

    elif(whichPlot == "velvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, quiver = "quiver")

    elif(whichPlot == "clustervideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, cluster = "cluster")

    elif(whichPlot == "ssvideosubset"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        firstIndex = int(sys.argv[7])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, subSet = "subset", firstIndex = firstIndex)

    elif(whichPlot == "ssvideonpt"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, npt = "npt")

    elif(whichPlot == "dpm"):
        plotDPMPacking(dirName, figureName, colorMap = True)

    elif(whichPlot == "dpmvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeDPMPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "comparedpm"):
        dirSample = dirName + os.sep + sys.argv[4]
        compareDPMPackings(dirName, dirSample, figureName)

    elif(whichPlot == "comparedpmvideo"):
        fileName = dirName + os.sep + sys.argv[4]
        dirName = dirName + os.sep + sys.argv[5]
        compareDPMPackingsVideo(dirName, fileName, figureName)

    elif(whichPlot == "dpmhop"):
        plotDPMcolorHOP(dirName, figureName)

    elif(whichPlot == "compvideo"):
        numFrames = int(sys.argv[4])
        makeCompressionVideo(dirName, figureName, numFrames)

    elif(whichPlot == "dpmrearrange"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeRearrengementsVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "spdpm"):
        plotSPDPMPacking(dirName, figureName)

    else:
        print("Please specify the type of plot you want")
