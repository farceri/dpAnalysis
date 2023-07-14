'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import itertools
import sys
import os
import utils
import shapeDescriptors as shape

def setAxis2D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setDPMAxes(boxSize, ax):
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxis2D(ax)

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

def plotDPMPacking(dirName, figureName, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, save = True, plot = True):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    area = np.array(np.loadtxt(dirName + os.sep + "areas.dat"))
    setDPMAxes(boxSize, ax)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotDeformableParticles(ax, pos, rad, nv, faceColor, edgeColor, colorMap, edgeColorMap, alpha)
    if(save == True):
        plt.savefig("/home/francesco/Pictures/dpm/packings/" + figureName + ".png", transparent=True, format = "png")
    if(plot == True):
        plt.show()
    else:
        return ax

def plotShapeDirectors(dirName, figureName, alpha = 0.2):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    # find largest eigenvector of moment of inertia
    eigs, eigv, pPos = shape.computeInertiaTensor(dirName, boxSize, nv, plot=False)
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pPos -= np.floor(pPos/boxSize) * boxSize
    eigvmax = np.zeros((numParticles, 2))
    eigvmin = np.zeros((numParticles, 2))
    for i in range(numParticles):
        eigvmax[i] = eigv[i,np.argmax(eigs[i])]
        eigvmin[i] = eigv[i,np.argmin(eigs[i])]
    ax = plotDPMPacking(dirName, figureName, alpha = alpha, save = False, plot = False)
    plt.savefig("/home/francesco/Pictures/dpm/packings/shape-packing-" + figureName + ".png", transparent=True, format = "png")
    ax.quiver(pPos[:,0], pPos[:,1], eigvmax[:,0], eigvmax[:,1], color='g', alpha=0.8)
    plt.savefig("/home/francesco/Pictures/dpm/packings/shape-dirmax-" + figureName + ".png", transparent=True, format = "png")
    ax.quiver(pPos[:,0], pPos[:,1], eigvmin[:,0], eigvmin[:,1], color='r', alpha=0.8)
    plt.savefig("/home/francesco/Pictures/dpm/packings/shape-dirboth-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotShapeAlignment(dirName, angleTh, figureName="directors", alpha = 0.6, plot=True):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    # find largest eigenvector of moment of inertia
    eigvmax, pPos = shape.getShapeDirections(dirName, boxSize, nv)
    intensity = shape.getVectorFieldAlignement(dirName, eigvmax, angleTh, 2, numParticles)
    colorList = [[1-x, 1-x, 1] for x in intensity]
    edgeColorList = np.ones((numParticles,3))
    edgeColorList[intensity==1] *= 0.2
    edgeColorList[intensity==0] *= 0.7
    ax = plotDPMPacking(dirName, figureName, colorList, edgeColorList, colorMap = True, edgeColorMap = True, alpha = alpha, save = False, plot = False)
    ax.quiver(pPos[intensity==0,0], pPos[intensity==0,1], eigvmax[intensity==0,0], eigvmax[intensity==0,1], color='g', alpha=0.2)
    ax.quiver(pPos[intensity==1,0], pPos[intensity==1,1], eigvmax[intensity==1,0], eigvmax[intensity==1,1], color='g', alpha=0.8)
    plt.savefig("/home/francesco/Pictures/dpm/packings/shape-alignment-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotShapeClusters(dirName, figureName, angleTh, alpha = 0.6, plot=True):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    eigvmax, pPos = shape.getShapeDirections(dirName, boxSize, nv)
    intensity = shape.getVectorFieldAlignement(dirName, eigvmax, angleTh, 2, numParticles)
    clusterList, particleLabel, clusterAngle = shape.clusterVectorField(dirName, eigvmax, intensity, angleTh, numParticles)
    # plot clusters
    clusterList = np.array(clusterList)
    colorLabel = cm.get_cmap('hsv', 90)
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    # plot background packing
    ax = plotDPMPacking(dirName, figureName, faceColor = [1,1,1], edgeColor = [0.8,0.8,0.8], colorMap = False, edgeColorMap = False, alpha = 0.4, save = False, plot = False)
    for i in range(clusterList.shape[0]):
        label = clusterList[i]
        if(particleLabel[particleLabel==label].shape[0]>3):
            #print(label, particleLabel[particleLabel==label].shape[0], np.argwhere(particleLabel==label))
            plotDeformableParticles(ax, pos, rad, nv, faceColor = colorLabel(clusterAngle[label]/90), clusterList=np.argwhere(particleLabel==label), lw=0.2)
            ax.quiver(pPos[particleLabel==label,0], pPos[particleLabel==label,1], eigvmax[particleLabel==label,0], eigvmax[particleLabel==label,1], color=colorLabel(clusterAngle[label]/90), alpha=0.8)
            plt.pause(0.5)
    plt.savefig("/home/francesco/Pictures/dpm/packings/shape-clusters-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def makeShapeClusterVideo(dirName, figureName, angleTh, numFrames = 20, firstStep = 1e07, freqPower = 4):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 200
    frames = []
    stepList = utils.getStepList(numFrames, firstStep, freqPower)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setDPMAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    rad = np.loadtxt(dirName + os.sep + "radii.dat")
    rad = np.array(rad)
    colorLabel = cm.get_cmap('viridis', 90)# based on orientation
    for i in stepList:
        highlightList = []
        pos = np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat")
        pos = np.array(pos)
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setDPMAxes(boxSize, axFrame)
        plotDeformableParticles(axFrame, pos, rad, nv, faceColor = [1,1,1], edgeColor = [0.8,0.8,0.8], colorMap = False, edgeColorMap = False, alpha = 0.4, highlightList=highlightList)
        #plot clusters
        eigvmax, pPos = shape.getShapeDirections(dirName + os.sep + "t" + str(i), boxSize, nv)
        intensity = shape.getVectorFieldAlignement(dirName + os.sep + "t" + str(i), eigvmax, angleTh, 2, numParticles)
        if(intensity[intensity==1].shape[0] == 0):
            print("no aligned particles at time step ", i)
        else:
            clusterList, particleLabel, clusterAngle = shape.clusterVectorField(dirName + os.sep + "t" + str(i), eigvmax, intensity, angleTh, numParticles)
            for c in range(clusterList.shape[0]):
                label = clusterList[c]
                if(particleLabel[particleLabel==label].shape[0]>3):
                    color = colorLabel(clusterAngle[label]/90)
                    plotDeformableParticles(axFrame, pos, rad, nv, faceColor = color, clusterList=np.argwhere(particleLabel==label), lw=0.2)
                    axFrame.quiver(pPos[particleLabel==label,0], pPos[particleLabel==label,1], eigvmax[particleLabel==label,0], eigvmax[particleLabel==label,1], color=color, alpha=0.8)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/cluster-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def stressShapeVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    contactdiff = utils.getContactDiff(dirName, numParticles, stepList)
    rearrangeId = np.argwhere(contactdiff==2)[1]
    print(rearrangeId)
    #numContactList = []
    shapeParam = []
    #eigMaxList = []
    #eigMinList = []
    eigMaxStressList = []
    for i in stepList:
        #contacts = np.loadtxt(dirName + os.sep + "t" + str(i) + "/neighbors.dat", dtype=int)+1
        #numContactList.append(np.argwhere(contacts[rearrangeId]>0).shape[0])
        area = np.loadtxt(dirName + os.sep + "t" + str(i) + "/areas.dat")
        perimeter = np.loadtxt(dirName + os.sep + "t" + str(i) + "/perimeters.dat")
        shapeParam.append(perimeter[rearrangeId]**2/(4*np.pi*area[rearrangeId]))
        #eigs, _, _ = shape.computeInertiaTensor(dirName + os.sep + "t" + str(i), boxSize, nv, plot=False)
        #eigMaxList.append(eigs[rearrangeId,1])
        #eigMinList.append(eigs[rearrangeId,0])
        eigsStress, _ = shape.computeStressTensor(dirName + os.sep + "t" + str(i), nv, plot=False)
        eigMaxStressList.append(eigsStress[rearrangeId,1])
    #numContactList = np.array(numContactList)
    shapeParam = np.array(shapeParam)
    #eigMaxList = np.array(eigMaxList)
    #eigMinList = np.array(eigMinList)
    eigMaxStressList = np.array(eigMaxStressList)
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    # animation
    #meanShape = np.mean(shapeParam)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5,5.5), dpi=150)
    frameTime = 200
    gcf = plt.gcf()
    def animate(i):
        ax[0].clear()
        ax[1].clear()
        #ax[0].plot(stepList, numContactList, linewidth=0, marker='o', color=[0.1,0.1,0.1], markersize=4)
        ax[1].plot(stepList, shapeParam, linewidth=1.2, color='k')
        #ax[1].plot(stepList, eigMaxList**2 - eigMinList**2, linewidth=1.2, color='k')
        #ax[1].plot(stepList, eigMaxList, linewidth=1.2, color='k')
        ax[0].plot(stepList, eigMaxStressList, linewidth=1.2, color='k')
        #ax[0].plot(np.ones(10)*stepList[i], np.linspace(-1, 10, 10), color='k', linestyle='--', linewidth=0.7)
        #ax[1].plot(np.ones(10)*stepList[i], np.linspace(-1, 10, 10), color='k', linestyle='--', linewidth=0.7)
        #ax[0].plot(stepList[i], numContactList[i], linewidth=0, marker='*', color=[1,0.2,0.2], markeredgecolor='k', markeredgewidth=0.5, markersize=15)
        ax[1].plot(stepList[i], shapeParam[i], linewidth=0, marker='*', color=[1,0.2,0.2], markeredgecolor='k', markeredgewidth=0.5, markersize=15)
        #ax[1].plot(stepList[i], eigMaxList[i]**2 - eigMinList[i]**2, marker='*', color=[1,0.2,0.2], markeredgecolor='k', markeredgewidth=0.5, markersize=15)
        #ax[1].plot(stepList[i], eigMaxList[i], linewidth=1.2, marker='*', color=[1,0.2,0.2], markeredgecolor='k', markeredgewidth=0.5, markersize=15)
        ax[0].plot(stepList[i], eigMaxStressList[i], linewidth=0, marker='*', color=[1,0.2,0.2], markeredgecolor='k', markeredgewidth=0.5, markersize=15)
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)
        ax[1].set_xlabel("$Time$ $fraction,$ $t/t_{relax}$", fontsize=17)
        ax[0].set_ylabel("$Stress,$ $s_{max}$", fontsize=17, labelpad=5)
        ax[1].set_ylabel("$Shape$ $parameter,$ $A$", fontsize=17, labelpad=5)
        ax[0].set_ylim(0.97, 1.42)
        #ax[0].set_ylim(-0.0002, 0.0016)
        #ax[1].set_ylim(-0.3, 3.3)
        ax[1].set_ylim(1.07, 1.33)
        #ax[1].set_xticks((0, 0.25, 0.5, 0.75, 1))
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        return gcf.artists
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/contactShape-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def checkShapeVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    shapeParam = []
    pos0 = np.loadtxt(dirName + "/positions.dat")
    _, perimeter0 = shape.getAreaAndPerimeterList(pos0, boxSize, nv)
    area0 = np.loadtxt(dirName + "/areas.dat")
    shapeParam0 = np.mean(perimeter0**2/(4*np.pi*area0))
    data = np.loadtxt(dirName + os.sep + "energy.dat")
    for i in stepList:
        pos = np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat")
        _, perimeter = shape.getAreaAndPerimeterList(pos, boxSize, nv)
        area = np.loadtxt(dirName + os.sep + "t" + str(i) + "/areas.dat")
        shapeParam.append(np.mean(perimeter**2/(4*np.pi*area)))
    # animation
    meanShape = np.mean(shapeParam)
    varShape = np.abs(shapeParam - meanShape)/meanShape
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    fig = plt.figure(figsize=(6.5,3.5), dpi=150)
    ax = plt.gca()
    frameTime = 200
    gcf = plt.gcf()
    def animate(i):
        ax.clear()
        ax.plot(stepList[:-1], varShape[:-1], linewidth=1, color='k')
        ax.plot(np.ones(10)*stepList[i], np.linspace(-1, 10, 10), color='r', linestyle='--', linewidth=1.2)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$time$ $fraction,$ $t/t_{relax}$", fontsize=17)
        ax.set_ylabel("$shape$ $fluctuation$", fontsize=17, labelpad = 15)
        ax.set_ylim(-0.02, 0.14)
        ax.set_yticks((0, 0.04, 0.08, 0.12))
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        return gcf.artists
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=True)
    anim.save("/home/francesco/Pictures/dpm/packings/checkShape-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeVelocityMapVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, numBins = 40, vertex = True):
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    xbin = np.linspace(0,boxSize[0],numBins)
    ybin = np.linspace(0,boxSize[1],numBins)
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    numVertices = np.sum(nv)
    # animation
    fig = plt.figure(dpi=150)
    ax = plt.gca()
    frameTime = 200
    gcf = plt.gcf()
    def animate(i):
        ax.clear()
        setDPMAxes(boxSize, ax)
        vmap = np.zeros((numBins-1, numBins-1))
        vel = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepList[i]) + os.sep + "velocities.dat"))
        if (vertex == True):
            pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepList[i]) + os.sep + "positions.dat"))
        else:
            pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepList[i]) + os.sep + "particlePos.dat"))
            vel = utils.computeParticleVelocities(vel, nv)
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        for id in range(pos.shape[0]):
            for x in range(numBins-1):
                if(pos[id,0] > xbin[x] and pos[id,0] < xbin[x+1]):
                    for y in range(numBins-1):
                        if(pos[id,1] > ybin[y] and pos[id,1] < ybin[y+1]):
                            vmap[x,y] += np.linalg.norm(vel[id])
        im = ax.pcolormesh(xbin, ybin, vmap, vmin = 0, vmax=0.015)
        #fig.colorbar(im)
        plt.tight_layout()
        return gcf.artists
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=True)
    anim.save("/home/francesco/Pictures/dpm/packings/velmap-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]
    figureName = sys.argv[3]

    if(whichPlot == "shape"):
        plotShapeDirectors(dirName, figureName)

    elif(whichPlot == "alignshape"):
        angleTh = float(sys.argv[4])
        plotShapeAlignment(dirName, figureName, angleTh)

    elif(whichPlot == "clustershape"):
        angleTh = float(sys.argv[4])
        plotShapeClusters(dirName, figureName, angleTh)

    elif(whichPlot == "clustervideo"):
        angleTh = float(sys.argv[4])
        numFrames = int(sys.argv[5])
        firstStep = float(sys.argv[6])
        stepFreq = float(sys.argv[7])
        makeShapeClusterVideo(dirName, figureName, angleTh, numFrames, firstStep, stepFreq)

    elif(whichPlot == "stressshape"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        stressShapeVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "checkshape"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        checkShapeVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "velmap"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        numBins = int(sys.argv[7])
        makeVelocityMapVideo(dirName, figureName, numFrames, firstStep, stepFreq, numBins)

    else:
        print("Please specify the type of plot you want")
