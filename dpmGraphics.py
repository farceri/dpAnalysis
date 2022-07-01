'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from scipy.optimize import curve_fit
import itertools
import sys
import os
import computeCorrelation
import shapeDescriptors

def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

def plotParticleCorr(ax, x, y, ylabel, color, legendLabel = None, logx = True, logy = False, linestyle = 'solid'):
    ax.plot(x, y, linewidth=1.5, color=color, linestyle = linestyle, label=legendLabel)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel(ylabel, fontsize=18)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')

def plotCorr(ax, x, y1, y2, ylabel, color, legendLabel = None, logx = True, logy = False):
    if(color == 'k'):
        ax.plot(x, y2, linewidth=1.2, color='k', marker='.')
        ax.plot(x, y1, linewidth=1.2, color='r', marker='.', label=legendLabel)
    else:
        #ax.plot(x, y2, linewidth=1, color=color, linestyle='--')
        ax.plot(x, y1, linewidth=1.5, color=color, label=legendLabel)
    ax.tick_params(axis='both', labelsize=17)
    #ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')

def plotDynamics(fileName, figureName):
    data = np.loadtxt(fileName)
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    # msd plot
    plotCorr(ax[0], data[1:,0], data[1:,1], data[1:,4], "$MSD$", color = 'k', logy = True)
    # isf plot
    plotCorr(ax[1], data[1:,0], data[1:,2], data[1:,5], "$ISF$", color = 'k')
    # chi plot
    #plotCorr(ax[1], data[1:,0], data[1:,3], data[1:,6], "$\\chi_4$", color = 'k')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig("/home/francesco/Pictures/dpm/" + figureName + ".png", transparent=False, format = "png")
    plt.show()

############################ check energy and force ############################
def plotEnergy(dirName):
    energy = np.loadtxt(dirName + os.sep + "energy.dat")
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    ax.plot(energy[:,0], energy[:,2], linewidth=1.5, color='k')
    ax.plot(energy[:,0], energy[:,3], linewidth=1.5, color='r', linestyle='--')
    ax.plot(energy[:,0], energy[:,2]+energy[:,3], linewidth=1.5, color='b', linestyle='dotted')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    #ax.set_ylabel("$Potential$ $energy$", fontsize=15)
    ax.set_ylabel("$Epot,$ $Ekin$", fontsize=15)
    plt.tight_layout()
    plt.show()

def plotEnergyScale(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    colorList = np.flip([[0.5,0,1], 'b', 'g', [0,1,0.5], 'y', [1,0.5,0], 'r', [1,0,0.5]])#, [0,0.5,1]
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    indexList = np.array([1,2,3,4,5,6,7])
    for i in indexList:
        if(os.path.exists(dirName + dataSetList[i] + "/ab/Dr1e-01-v03e-03/dynamics-test/")):
            data = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v03e-03/dynamics-test/energy.dat")
            nv = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v03e-03/dynamics-test/numVertexInParticleList.dat")
            numVertices = np.sum(nv)
            data = data[1:,:]
            ax.plot(data[:,0], numVertices*data[:,3]/data[:,2], linewidth=1.5, color=colorList[i], marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $step$", fontsize=17)
    ax.set_ylabel("$T_{eff}/e_{pot}$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/scale-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def compareEnergy(dirName1, dirName2, figureName):
    energy1 = np.loadtxt(dirName1 + os.sep + "energy.dat")
    energy2 = np.loadtxt(dirName2 + os.sep + "energy.dat")
    print(energy1.shape, energy2.shape)
    #energy1 = energy1[:10000,:]
    #energy2 = energy2[:10000,:]
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.plot(energy1[:,1], energy1[:,2], linewidth=2.5, color='k')
    ax.plot(energy2[:,1], energy2[:,2], linewidth=1.5, color='y')
    ax.plot(energy1[:,1], energy1[:,3], linewidth=2.5, color='k', linestyle = '--')
    ax.plot(energy2[:,1], energy2[:,3], linewidth=1.5, color='y', linestyle = '--')
    ax.plot(energy1[:,1], energy1[:,2]+energy1[:,3], linewidth=2.5, color='k', linestyle = 'dotted')
    ax.plot(energy2[:,1], energy2[:,2]+energy2[:,3], linewidth=1.5, color='y', linestyle = 'dotted')
    #ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    #ax.set_ylabel("$Potential$ $energy$", fontsize=15)
    ax.set_ylabel("$Epot,$ $Ekin$", fontsize=15)
    plt.tight_layout()
    ax.legend(("$NVE$", "$Active Brownian$"), loc = (0.62,0.6), fontsize = 12)
    plt.savefig("/home/francesco/Pictures/dpm/energy" + figureName + ".png", transparent=False, format = "png")
    fig2 = plt.figure(1, dpi = 120)
    ax2 = fig2.gca();
    #ax2.plot(energy1[:,1], energy1[:,3]+energy1[:,2]-energy2[:,3]-energy2[:,2], color='k', linewidth=1.2)
    ax2.plot(energy1[:,1], energy1[:,4]/987, color='k', linewidth=2.2)
    ax2.plot(energy2[:,1], energy2[:,4]/987, color='y', linewidth=1.2)
    print("average difference: ", np.mean(energy1[:,2] - energy2[:,2]), np.sqrt(np.mean((energy1[:,2] - energy2[:,2])**2)))
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax2.set_ylabel("$Force$ $difference$", fontsize=15)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/force" + figureName + ".png", transparent=False, format = "png")
    plt.show()

# this only works for one deformable particle
def compareForces(dirName, fileName):
    # compare forces between cell format and dir format as a function of time
    maxStep = 101
    fig = plt.figure(dpi=150)
    ax = fig.gca()
    colorList = cm.get_cmap('viridis', maxStep)
    for i in range(1, maxStep, 10):
        # first read from directory
        f1 = np.loadtxt(dirName + os.sep + "forces_step" + str(i) + ".dat")
        # then read forces from cell format file
        f2 = np.loadtxt(fileName, skiprows=(32*i), max_rows = 32, usecols = (5,6))
        ax.plot(f1[:,0], f1[:,1], color = colorList(i/maxStep), marker = 'o', markersize = 4, linewidth = 0)
        ax.plot(f2[:,0], f2[:,1], color = 'k', marker = 'o', markersize = 4, linewidth = 0)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("$fx$", fontsize=15)
        ax.set_ylabel("$fy$", fontsize=15)
        plt.pause(0.2)
        plt.tight_layout()
    plt.show()

def compareForcesVideo(dirName, fileName, figureName):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    # compare forces between cell format and dir format as a function of time
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
    colorList = cm.get_cmap('viridis', maxStep)
    f1 = np.loadtxt(dirName + os.sep + "forces_step" + str(maxStep-1) + ".dat")
    # then read forces from cell format file
    f2 = np.loadtxt(fileName, skiprows=(32*(maxStep-1)), max_rows = 32, usecols = (5,6))
    xBounds = [np.min(f1[:,0]-f2[:,0]), np.max(f1[:,0]-f2[:,0])]
    yBounds = [np.min(f1[:,1]-f2[:,1]), np.max(f1[:,1]-f2[:,1])]
    print(xBounds, yBounds)
    for i in range(1, maxStep, 10):
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        # first read from directory
        f1 = np.loadtxt(dirName + os.sep + "forces_step" + str(i) + ".dat")
        # then read forces from cell format file
        f2 = np.loadtxt(fileName, skiprows=(32*i), max_rows = 32, usecols = (5,6))
        axFrame.plot(f1[:,0]-f2[:,0], f1[:,1]-f2[:,1], color = colorList(i/maxStep), marker = 'o', markersize = 4, linewidth = 0)
        #axFrame.plot(f2[:,0], f2[:,1], color = 'k', marker = 'o', markersize = 4, linewidth = 0)
        axFrame.tick_params(axis='both', labelsize=12)
        axFrame.set_xlabel("$fx$", fontsize=15)
        axFrame.set_ylabel("$fy$", fontsize=15)
        axFrame.set_xlim(100*xBounds[0], 100*xBounds[1])
        axFrame.set_ylim(100*yBounds[0], 100*yBounds[1])
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)


########################### check and plot compression #########################
def plotHOP(dirName, figureName):
    step = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for dir in os.listdir(dirName)[::10]:
        if(os.path.isdir(dirName + os.sep + dir)):
            step.append(float(dir[1:]))
            psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + dir)
            hop.append(np.mean(psi6))
            err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    step = np.array(step)
    hop = np.array(hop)
    err = np.array(err)
    hop = hop[np.argsort(step)]
    err = err[np.argsort(step)]
    step = np.sort(step)
    plotErrorBar(ax, step, hop, err, "$simulation$ $step$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/dpm/hexatic-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotHOPVSphi(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    phi = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        phi.append(computeCorrelation.readFromParams(dirName + os.sep + dataSetList[i], "phi"))
        psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop.append(np.mean(psi6))
        err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    plotErrorBar(ax, phi, hop, err, "$packing$ $fraction,$ $\\varphi$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/dpm/hop-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotHOPAlongCompression(dirName, figureName):
    dataSetList = np.array(os.listdir(dirName))
    phi = dataSetList.astype(float)
    dataSetList = dataSetList[np.argsort(phi)]
    phi = np.sort(phi)
    hop = np.zeros(phi.shape[0])
    err = np.zeros(phi.shape[0])
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop[i] = np.mean(psi6)
        err[i] = np.sqrt(np.var(psi6)/psi6.shape[0])
    ax.errorbar(phi[hop>0], hop[hop>0], err[hop>0], marker='o', color='k', markersize=5, markeredgecolor='k', markeredgewidth=0.7, linewidth=1, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$hexatic$ $order$ $parameter,$ $\\psi_6$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/hop-comp-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotPSI6P2(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    stepList = getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    hop = []
    p2 = []
    for i in stepList:
        psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + "t" + str(i), boxSize)
        hop.append(np.mean(psi6))
        eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + "t" + str(i), boxSize, nv)
        angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
        p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    stepList -= stepList[0]
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,5), dpi=150)
    ax[0].plot(stepList, hop, linewidth=1.2, color='b')
    ax[1].plot(stepList, p2, linewidth=1.2, color='g')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$time$ $fraction,$ $t/t_{relax}$", fontsize=17)
    ax[0].set_ylabel("$\\langle \\psi_6 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$\\langle P_2 \\rangle$", fontsize=17, labelpad=-5)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/psi6-p2-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotVelOrder(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, distTh = 0.1):
    stepList = getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    velcontact = []
    veldistance = []
    for i in stepList:
        stepVelcorr = computeCorrelation.computeVelCorrContact(dirName + os.sep + "t" + str(i), nv)
        velcontact.append(np.mean(stepVelcorr))
        stepVelcorr = computeCorrelation.computeVelCorrDistance(dirName + os.sep + "t" + str(i), boxSize, nv, distTh)
        veldistance.append(np.mean(stepVelcorr))
    stepList -= stepList[0]
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    fig = plt.subplots(figsize=(6,3), dpi=150)
    ax = plt.gca()
    ax.plot(stepList, velcontact, linewidth=1.2, color='k')
    ax.plot(stepList, veldistance, linewidth=1.2, color='r')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$time$ $fraction,$ $t/t_{relax}$", fontsize=17)
    ax.set_ylabel("$\\langle \\sum_j \\hat{v}_i \\cdot \\hat{v}_j \\rangle$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/velcontact-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotVelCorrSpace(dirName, figureName):
    distThList = np.linspace(0.12,0.8,30)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    velcorr = []
    velcontact = np.mean(computeCorrelation.computeVelCorrContact(dirName + os.sep + "t100000", nv))
    for distTh in distThList:
        stepVelcorr = computeCorrelation.computeVelCorrDistance(dirName + os.sep + "t100000", boxSize, nv, distTh)
        velcorr.append(np.mean(stepVelcorr))
    fig = plt.subplots(dpi=150)
    ax = plt.gca()
    ax.plot(distThList, velcorr, linewidth=1.2, color='b')
    ax.plot(0.125, velcontact, markersize=18, color='r', markeredgecolor='k', markeredgewidth=0.5, marker="*")
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Distance,$ $\\Delta r$", fontsize=17)
    ax.set_ylabel("$\\langle \\sum_j \\hat{v}_i \\cdot \\hat{v}_j \\rangle_{|\\vec{r}_i - \\vec{r}_j| = \\Delta r}$", fontsize=17)
    #ax.set_ylim(-0.038,0.134)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/velcorrspace-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleCompression(dirName, figureName, compute = "compute"):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    if(compute=="compute"):
        phi = []
        pressure = []
        hop = []
        zeta = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir)):
                phi.append(computeCorrelation.readFromParams(dirName + os.sep + dir, "phi"))
                pressure.append(computeCorrelation.readFromParams(dirName + os.sep + dir, "pressure"))
                boxSize = np.loadtxt(dirName + os.sep + dir + "/boxSize.dat")
                psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + dir, boxSize)
                hop.append(np.mean(psi6))
                contacts = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "contacts.dat"))
                z = 0
                if(contacts.shape[0] != 0):
                    for p in range(contacts.shape[0]):
                        z += np.sum(contacts[p]>-1)
                    zeta.append(z/contacts.shape[0])
                else:
                    zeta.append(0)
        pressure = np.array(pressure)
        hop = np.array(hop)
        zeta = np.array(zeta)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        hop = hop[np.argsort(phi)]
        zeta = zeta[np.argsort(phi)]
        phi = np.sort(phi)
        np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, hop, zeta)))
    else:
        data = np.loadtxt(dirName + os.sep + "compression.dat")
        phi = data[:,0]
        pressure = data[:,1]
        hop = data[:,2]
        zeta = data[:,3]
    ax[0].semilogy(phi, pressure, color='k', linewidth=1.5)
    ax[1].plot(phi, zeta, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$pressure,$ $p$", fontsize=17)
    #ax[0].set_ylabel("$hexatic$ $order,$ $\\psi_6$", fontsize=17)
    ax[1].set_ylabel("$coordination$ $number,$ $z$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/comp-control-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotCompression(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    pressure = []
    zeta = []
    phi = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            phi.append(computeCorrelation.readFromParams(dirName + os.sep + dir, "phi"))
            pressure.append(computeCorrelation.readFromParams(dirName + os.sep + dir, "pressure"))
            contacts = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "neighbors.dat"))
            z = 0
            if(contacts.shape[0] != 0):
                for p in range(contacts.shape[0]):
                    z += np.sum(contacts[p]>-1)
                zeta.append(z/contacts.shape[0])
            else:
                zeta.append(0)
    pressure = np.array(pressure)
    zeta = np.array(zeta)
    phi = np.array(phi)
    pressure = pressure[np.argsort(phi)]
    zeta = zeta[np.argsort(phi)]
    phi = np.sort(phi)
    np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, zeta)))
    ax[0].plot(phi, pressure, color='k', linewidth=1.5)
    ax[1].plot(phi, zeta, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$pressure,$ $p$", fontsize=17)
    ax[1].set_ylabel("$coordination$ $number,$ $z$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/comp-control-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotCompressionPSI6P2(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    phi = []
    hop = []
    p2 = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            phi.append(computeCorrelation.readFromParams(dirName + os.sep + dir, "phi"))
            boxSize = np.loadtxt(dirName + os.sep + dir + "/boxSize.dat")
            nv = np.loadtxt(dirName + os.sep + dir + "/numVertexInParticleList.dat", dtype=int)
            psi6 = computeCorrelation.computeHexaticOrder(dirName + os.sep + dir, boxSize)
            hop.append(np.mean(psi6))
            eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + dir, boxSize, nv)
            angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
            p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    phi = np.array(phi)
    hop = np.array(hop)
    p2 = np.array(p2)
    hop = hop[np.argsort(phi)]
    p2 = p2[np.argsort(phi)]
    phi = np.sort(phi)
    hop = hop[phi>0.65]
    p2 = p2[phi>0.65]
    phi = phi[phi>0.65]
    np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, zeta)))
    ax[0].plot(phi, p2, color='k', linewidth=1.5)
    ax[1].plot(phi, hop, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$nematic$ $order,$ $\\langle p2 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$hexagonal$ $order,$ $\\langle \\psi_6 \\rangle$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/dpm/comp-param-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotCompressionSet(dirName, figureName):
    #dataSetList = np.array(["kb1e-03", "kb1e-02", "kb1e-01", "kb2e-01", "kb4e-01", "kb5e-01-kakl", "kb6e-01", "kb8e-01"])
    dataSetList = np.array(["A1_1-sigma17", "A1_2-sigma17", "A1_3-sigma17"])
    phiJ = np.array([0.8301, 0.8526, 0.8242, 0.8205, 0.8176, 0.7785, 0.7722, 0.7707])
    colorList = ['k', [0.5,0,1], 'b', 'g', [0.8,0.9,0.2], [1,0.5,0], 'r', [1,0,0.5]]
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        pressure = []
        phi = []
        for dir in os.listdir(dirName + dataSetList[i]):
            if(os.path.isdir(dirName + dataSetList[i] + os.sep + dir)):
                phi.append(computeCorrelation.readFromParams(dirName + dataSetList[i] + os.sep + dir, "phi"))
                pressure.append(computeCorrelation.readFromParams(dirName + dataSetList[i] + os.sep + dir, "pressure"))
        pressure = np.array(pressure)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        phi = np.sort(phi)
        phi = phi[pressure>0]
        pressure = pressure[pressure>0]
        np.savetxt(dirName + dataSetList[i] + os.sep + "compression.dat", np.column_stack((phi, pressure)))
        ax.semilogy(phi, pressure, color=colorList[i], linewidth=1.2, label=dataSetList[i])
    ax.legend(loc = "upper left", fontsize = 12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$pressure,$ $p$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/compression-" + figureName + ".png", transparent=False, format = "png")
    plt.show()


########################## plot correlation functions ##########################
def plotParticleEnergyScale(dirName, sampleName, figureName):
    Dr = []
    T = []
    pressure = []
    timeStep = 3e-04
    dataSetList = np.array(["1e03", "1e02", "1e01", "1", "1e-01", "1e-02", "1e-03", "1e-04"])
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    data = np.loadtxt(dirName + "../../T1/energy.dat")
    ax[0].semilogx(1/timeStep, np.mean(data[10:,4]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    ax[1].semilogx(1/timeStep, np.mean(data[10:,6]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/")):
            data = np.loadtxt(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/energy.dat")
            Dr.append(float(dataSetList[i]))
            T.append(np.mean(data[10:,4]))
            pressure.append(np.mean(data[10:,6]))
    ax[0].tick_params(axis='both', labelsize=15)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[0].semilogx(Dr, T, linewidth=1.2, color='k', marker='o')
    ax[1].semilogx(Dr, pressure, linewidth=1.2, color='k', marker='o')
    ax[0].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[1].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[0].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[1].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[0].set_ylabel("$Temperature,$ $T$", fontsize=18)
    ax[1].set_ylabel("$Pressure,$, $p$", fontsize=18)
    ax[0].set_ylim(0.98,3.8)#1.15)#
    ax[1].set_ylim(5e-05,6.4e-03)#9.6e-04)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/soft-Tp-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleFDT(dirName, figureName, Dr, driving):
    tmeasure = 100
    fextStr = np.array(["2", "3", "4"])
    fext = fextStr.astype(float)
    mu = np.zeros((fextStr.shape[0],2))
    T = np.zeros((fextStr.shape[0],2))
    fig, ax = plt.subplots(1, 2, figsize = (12.5, 5), dpi = 120)
    corr = np.loadtxt(dirName + os.sep + "dynamics/corr-log-q1.dat")
    timeStep = computeCorrelation.readFromParams(dirName + os.sep + "dynamics", "dt")
    diff = np.mean(corr[corr[:,0]*timeStep>tmeasure,1]/(2*corr[corr[:,0]*timeStep>tmeasure,0]*timeStep))
    for i in range(fextStr.shape[0]):
        sus = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/susceptibility.dat")
        sus = sus[sus[:,0]>tmeasure,:]
        mu[i,0] = np.mean(sus[:,1]/sus[:,0])
        mu[i,1] = np.std(sus[:,1]/sus[:,0])
        energy = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/energy.dat")
        energy = energy[energy[:,0]>tmeasure,:]
        T[i,0] = np.mean(energy[:,4])
        T[i,1] = np.std(energy[:,4])
    ax[0].errorbar(fext, mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, T[:,0], T[:,1], color='b', marker='D', fillstyle='none', markeredgecolor = 'b', markeredgewidth = 1.5, markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, diff/mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    for i in range(ax.shape[0]):
        ax[i].tick_params(axis='both', labelsize=15)
        ax[i].set_xlabel("$f_0$", fontsize=18)
    ax[0].set_ylabel("$Mobility,$ $\\chi / t = \\mu$", fontsize=18)
    ax[1].set_ylabel("$Temperature$", fontsize=18)
    ax[1].legend(("$Kinetic,$ $T$", "$FDT,$ $D/ \\mu = T_{FDT}$"), loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    #fig.savefig("/home/francesco/Pictures/soft/pFDT-" + figureName + ".png", transparent=True, format = "png")
    pSize = 2 * np.mean(np.array(np.loadtxt(dirName + os.sep + "dynamics/particleRad.dat")))
    Pe = pSize * driving / 1e-02
    Pev = ((driving / 1e03) / Dr) / pSize
    print("Pe: ", Pev, " susceptibility: ",  np.mean(mu[i,0]), " diffusivity: ", diff, " T_FDT: ", diff/np.mean(mu[i,0]))
    np.savetxt(dirName + "FDTtemp.dat", np.column_stack((Pe, Pev, np.mean(T[:,0]), np.std(T[:,0]), np.mean(mu[:,0]), np.std(mu[:,0]), diff)))
    plt.show()

def plotParticleTeff(dirName, sampleName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    mu = np.zeros((dataSetList.shape[0],2))
    temp = np.zeros((dataSetList.shape[0],2))
    tau = np.zeros(dataSetList.shape[0])
    diff = np.zeros(dataSetList.shape[0])
    phi = np.zeros(dataSetList.shape[0])
    for i in range(dataSetList.shape[0]):
        difftau = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")
        diff[i] = difftau[-1,-1]
        tau[i] = difftau[-1,-2]
        mufext = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/muFext.dat")
        mu[i,0] = np.mean(mufext[:,2])
        mu[i,1] = np.std(mufext[:,2])
        energy = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/energy.dat")
        temp[i,0] = np.mean(energy[:,4])
        temp[i,1] = np.std(energy[:,4])
        phi[i] = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/dynamics", "phi")
    np.savetxt(dirName + "/DFT.dat", np.column_stack((phi, tau, diff, mu, temp)))
    fig, ax = plt.subplots(dpi = 120)
    ax.errorbar(phi, temp[:,0], temp[:,1], color='b', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    ax.errorbar(phi, diff/mu[:,0], mu[:,1], color='k', marker='v', markersize=8, markeredgewidth=2, lw=1, ls='--', capsize=3, fillstyle='none')
    ax.set_ylim(0.06,0.66)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=18)
    ax.set_ylabel("$Temperature,$ $T_K$ $T_{FDT}$", fontsize=18)
    ax.legend(("$Kinetic,$ $T$", "$FDT,$ $D/ \\mu = T_{FDT}$"), loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/Teff-" + figureName + ".png", transparent=True, format = "png")
    fig1, ax1 = plt.subplots(dpi = 120)
    ax1.errorbar(phi, mu[:,0], mu[:,1], color='r', marker='s', markersize=8, lw=0.5, capsize=3)
    ax1.semilogy(phi, diff, color='g', marker='v', markersize=8, lw=0.5)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=18)
    ax1.legend(("$Diffusivity,$ $D$", "$Mobility,$ $\\mu$"), loc="lower left", fontsize=12)
    fig1.tight_layout()
    fig1.savefig("/home/francesco/Pictures/soft/diffmu-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleDiffusivity(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    Tlist = np.array(["1e-01", "8e-02", "6e-02", "4e-02", "2e-02", "1e-02"])
    colorList = cm.get_cmap('plasma', Tlist.shape[0])
    fig, ax = plt.subplots(dpi = 120)
    for t in Tlist.shape[0]:
        tau = np.zeros(dataSetList.shape[0])
        diff = np.zeros(dataSetList.shape[0])
        phi = np.zeros(dataSetList.shape[0])
        for i in range(dataSetList.shape[0]):
            difftau = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + Tlist[t] + "/tauDiff.dat")
            diff[i] = difftau[-1,-1]
            tau[i] = difftau[-1,-2]
            phi[i] = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/active-langevin/T" + Tlist[t] + "/dynamics", "phi")
        np.savetxt(dirName + "/diffusivity.dat", np.column_stack((phi, tau, diff)))
        ax.semilogy(phi, diff, color=colorList(t/Tlist.shape[0]), marker='v', markersize=8, lw=0.5)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=18)
    ax.set_ylabel("$Diffusivity,$ $D$", fontsize=18)
    fig.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/diff-vsT-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def computeTau(data):
    relStep = np.argwhere(data[:,2]>0.4)[-1,0]
    if(relStep + 1 < data.shape[0]):
        t1 = data[relStep,0]
        t2 = data[relStep+1,0]
        ISF1 = data[relStep,2]
        ISF2 = data[relStep+1,2]
        slope = (ISF2 - ISF1)/(t2 - t1)
        intercept = ISF2 - slope * t2
        return (np.exp(-1) - intercept)/slope
    else:
        return data[relStep,0]

def plotParticleDynamics(dirName, sampleName, figureName):
    dataSetList = np.array(["1e01", "2e01", "4e01", "6e01", "8e01", "1e02"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(dpi = 150)
    # plot brownian dynamics
    data = np.loadtxt(dirName + "../../langevin/T1e-02/dynamics/corr-log-q1.dat")
    timeStep = computeCorrelation.readFromParams(dirName + "../../langevin/T1e-02/dynamics/", "dt")
    ax.semilogx(data[1:,0]*timeStep, data[1:,2], color='g', linestyle='--', linewidth=1.2, markersize = 10, markeredgewidth = 0.2, label = "passive")
    #ax.semilogx(data[1:,0]*timeStep, data[1:,1]/(data[1:,0]*timeStep), color='g', linestyle='--', linewidth=1.2, markersize = 10, markeredgewidth = 0.2, label = "passive")
    # plot all the active dynamics
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + sampleName + "-f0" + dataSetList[i] + "/dynamics/")):
            timeStep = computeCorrelation.readFromParams(dirName + "/Dr" + sampleName + "-f0" + dataSetList[i] + "/dynamics/", "dt")
            data = np.loadtxt(dirName + "/Dr" + sampleName + "-f0" + dataSetList[i] + "/dynamics/corr-log-q1.dat")
            print("diffusivity: " , np.mean(data[data[:,0]*timeStep>100,1]/(4*data[data[:,0]*timeStep>100,0]*timeStep)))
            print("relaxation time: ", timeStep*computeTau(data))
            legendlabel = "$D_r=$" + sampleName + ", $f_0=$" + dataSetList[i]
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((i)/dataSetList.shape[0]), legendLabel = legendlabel)
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList((i)/dataSetList.shape[0]), legendLabel = legendlabel, logy=True)
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1]/(data[1:,0]*timeStep), "$MSD(\\Delta t) / \\Delta t$", color = colorList((i)/dataSetList.shape[0]), legendLabel = legendlabel, logy = True)
    #ax.plot(np.linspace(1e-05,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.2, color='k')
    ax.legend(fontsize=10, loc="lower left")
    #ax.legend(loc = "upper right", fontsize = 11)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    #ax.set_xlim(2e-04, 4e02)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/p" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareParticleDynamicsFDT(dirName, sampleName, figureName):
    f0SetList = np.array(["1e01", "2e01", "4e01", "6e01", "8e01", "1e02"])
    fdtSetList = np.array(["0.02", "0.08", "0.15", "0.21", "0.31", "0.48"])#0.21
    colorList = cm.get_cmap('plasma', f0SetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(f0SetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + sampleName + "-f0" + f0SetList[i] + "/dynamics/")):
            data = np.loadtxt(dirName + "/Dr" + sampleName + "-f0" + f0SetList[i] + "/dynamics/corr-log-q1.dat")
            timeStep = computeCorrelation.readFromParams(dirName + "/Dr" + sampleName + "-f0" + f0SetList[i] + "/dynamics/", "dt")
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = 'log')
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/f0SetList.shape[0]))
            data = np.loadtxt(dirName + "../../../glassFDT/langevin/T" + fdtSetList[i] + "/dynamics/corr-log-q1.dat")
            timeStep = computeCorrelation.readFromParams(dirName + "../../../glassFDT/langevin/T" + fdtSetList[i] + "/dynamics/", "dt")
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = 'log')
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", linestyle='--', color = colorList(i/f0SetList.shape[0]))
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    #ax.set_xlim(4e-04, 4e07)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcompare-active-thermalFDT-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleDensity(dirName, sampleName, numBins, figureName):
    dataSetList = np.array(["1e01", "2e01", "4e01", "6e01", "8e01", "1e02"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0]+1)
    fig, ax = plt.subplots(dpi = 120)
    data = np.loadtxt(dirName + "../../langevin/T1e-02/dynamics/localDensity-N" + numBins + ".dat")
    data = data[data[:,1]>0]
    ax.plot(data[1:,0], data[1:,1], linewidth=1.2, marker='s', markersize=8, color='g')
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + sampleName + "-f0" + dataSetList[i] + "/dynamics/")):
            data = np.loadtxt(dirName + "/Dr" + sampleName + "-f0" + dataSetList[i] + "/dynamics/localDensity-N" + numBins + ".dat")
            data = data[data[:,1]>0]
            ax.plot(data[1:,0], data[1:,1], linewidth=1.2, marker='o', markersize=4, color=colorList((i)/dataSetList.shape[0]))
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('$P(\\varphi)$', fontsize=18)
    ax.set_xlabel('$\\varphi$', fontsize=18)
    ax.set_yscale('log')
    #ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/densityPDF-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotParticleDynamicsVStemp(dirName, figureName):
    T = []
    Deff = []
    tau = []
    dataSetList = np.array(["1e-01", "8e-02", "6e-02"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q1.dat")
            timeStep = computeCorrelation.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "dt")
            energy = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/energy.dat")
            T.append(np.mean(energy[10:,4]))
            Deff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*computeTau(data))
            #tau.append(0.5*(data[relStep,0]+data[relStep+1,0])*timeStep)
            #print("T: ", T[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep)
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = 'log')
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    #ax.set_xlim(4e-04, 4e07)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/pcorr-T-" + figureName + ".png", transparent=True, format = "png")
    T = np.array(T)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    #ax[1].semilogy(phi, Deff, linewidth=1.5, color='k', marker='o')
    ax.loglog(1/T, np.log(tau), linewidth=1.5, color='k', marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    #ax[1].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
    ax.set_ylabel("$log(\\tau)$", fontsize=17)
    plt.tight_layout()
    np.savetxt(dirName + "../diff-tau-vs-temp.dat", np.column_stack((T, Deff, tau)))
    plt.savefig("/home/francesco/Pictures/soft/ptau-T-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleTauVStemp(dirName, figureName):
    phiJ = 0.84567
    mu = 1.5
    delta = 1.05
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/diff-tau-vs-temp.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/diff-tau-vs-temp.dat")
            phi = computeCorrelation.readFromParams(dirName + dataSetList[i], "phi")
            #ax.loglog(1/data[:,0], np.log(data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
            ax.loglog(np.abs(phi - phiJ)**(2/mu)/data[:,0], np.abs(phiJ - phi)**(delta) * np.log(np.sqrt(data[:,0])*data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
    ax.set_xlim(2.6e-03, 4.2e02)
    ax.set_ylim(6.3e-03, 1.2)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    ax.set_xlabel("$|\\varphi - \\varphi_J|^{2/\\mu}/T$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax.set_ylabel("$|\\varphi - \\varphi_J|^\\delta \\log(\\tau T^{1/2})$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-vsT-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotParticleTau(dirName, sampleName, figureName):
    dataSetList = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")
            data = data[1:,:]
            ax.loglog(data[data[:,4]>0,1], data[data[:,4]>0,4], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    #ax.set_xlim(1.3, 15300)
    ax.plot(np.linspace(5,100,50), 2e04/np.linspace(5,100,50)**2, linestyle='--', linewidth=1.5, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Wave$ $vector$ $magnitude,$ $q$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotParticleDynamicsVSQ(dirName, figureName):
    dataSetList = np.array(["1", "2", "3", "5", "10", "20", "30", "50", "100"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/corr-log-q" + dataSetList[i] + ".dat")):
            data = np.loadtxt(dirName + "/corr-log-q" + dataSetList[i] + ".dat")
            timeStep = computeCorrelation.readFromParams(dirName, "dt")
            legendlabel = "$q=2\\pi/($" + dataSetList[i] + "$\\times d)$"
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.legend(loc = "lower left", fontsize = 12)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-q-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleTauVsactivity(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/tauDiff.dat")
            ax.loglog(1/data[:,1], data[:,2], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    #ax[1].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
    ax.set_ylabel("$log(\\tau)$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-active-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleDynamicsVSphi(dirName, sampleName, figureName):
    phi = []
    tau = []
    Deff = []
    dirDyn = "/langevin/"
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(dpi = 150)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + dirDyn  + "/T" + sampleName + "/dynamics/corr-log-q1.dat")
            timeStep = computeCorrelation.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "dt")
            phi.append(computeCorrelation.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "phi"))
            Deff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*computeTau(data))
            print("phi: ", phi[-1], " Deff: ", Deff[-1], " tau: ", tau[-1])
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel, logy = True)
            #plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel)
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    #ax.set_ylim(3e-06,37100)#2.3e-04
    ax.legend(loc = "upper left", fontsize = 11, ncol = 2)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-vsphi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotParticleISFVSphi(dirName, sampleName, figureName):
    def func(x, a, b, c):
        return a * np.exp(-x**b/c)
    phi = []
    tau = []
    dirDyn = "/langevin/"
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(dpi = 150)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + dirDyn  + "/T" + sampleName + "/dynamics/corr-log-q1.dat")
            timeStep = computeCorrelation.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "dt")
            phi.append(computeCorrelation.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "phi"))
            tau.append(timeStep*computeTau(data))
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            plotParticleCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel)
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    #ax.set_ylim(3e-06,37100)#2.3e-04
    ax.legend(loc = "lower left", fontsize = 11, ncol = 2)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-vsphi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def checkDynamics(dirName, figureName):
    data = np.loadtxt(dirName + os.sep + "energy.dat")
    fig, ax = plt.subplots(1, 3, figsize = (14, 4), dpi = 120)
    # potential energy
    ax[0].plot(data[1:,0], data[1:,2], linewidth=0.5, color='k', marker='o', markersize=3)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$E_{pot}$", fontsize=15)
    # temperature
    T = np.mean(data[1:,3])
    ax[1].plot(data[1:,0], np.sqrt((data[1:,3] - T)**2)/T, linewidth=0.5, color='k', marker='o', markersize=3)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_ylabel("$|T-\\langle T \\rangle | / \\langle T \\rangle$", fontsize=15)
    # density
    phi = np.mean(data[1:,4])
    ax[2].plot(data[1:,0], np.sqrt((data[1:,4] - phi)**2)/phi, linewidth=0.5, color='k', marker='o', markersize=3)
    ax[2].tick_params(axis='both', labelsize=12)
    ax[2].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[2].set_ylabel("$|\\varphi - \\langle \\varphi \\rangle | / \\langle \\varphi \\rangle$", fontsize=15)
    plt.tight_layout()
    print("average temperature: ", T)
    plt.savefig("/home/francesco/Pictures/dpm/check-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDynamicsVSphi(dirName, v0String, figureName, plotDiff = True, save = False):
    D0 = float(v0String) **2 / (2 * 1e-01)
    phi = []
    Teff = []
    Deff = []
    vDeff = []
    tau = []
    vtau = []
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 9), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")):
            timeStep = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "dt")
            currentPhi = computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "phi")
            data = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")
            if(data[-1,2]<0.2):
                Teff.append(computeCorrelation.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "temperature"))
                phi.append(currentPhi)
                Deff.append(data[data[:,2]>np.exp(-1),1][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                vDeff.append(data[data[:,2]>np.exp(-1),4][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                tau.append(data[data[:,2]>np.exp(-1),0][-1]*timeStep)
                vtau.append(data[data[:,5]>np.exp(-1),0][-1]*timeStep)
                print("phi: ", phi[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep, " timeStep:", timeStep)
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            plotCorr(ax[0], data[1:,0]*timeStep, data[1:,1], data[1:,4], "$MSD(\\Delta t)$", colorList(i/dataSetList.shape[0]), logy=True)
            plotCorr(ax[1], data[1:,0]*timeStep, data[1:,2], data[1:,5], "$ISF(\\Delta t)$", colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
            #plotCorr(ax[1], data[1:,0], data[1:,3], data[1:,6], "$\\chi_4$", colorList(i/dataSetList.shape[0]))
    ax[1].plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax[1].legend(loc = "lower left", fontsize = 12)
    ax[1].set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=20)
    ax[0].set_ylim(8e-15,2)
    ax[1].set_xlim(7e-03, 2e08)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if(save=="save"):
        plt.savefig("/home/francesco/Pictures/dpm/corr-phi-" + figureName + ".png", transparent=False, format = "png")
    np.savetxt(dirName + "diff-tau-vs-phi-v0" + v0String + ".dat", np.column_stack((phi, Teff, Deff, tau, vDeff, vtau)))
    if(plotDiff == True):
        fig, ax = plt.subplots(1, 2, figsize = (13, 5), dpi = 120)
        phi = np.array(phi)
        ax[0].semilogy(phi, Deff, linewidth=1.5, color='k', marker='o')
        #ax[0].semilogy(phi, vDeff, linewidth=1.5, color='k', marker='o', linestyle='--')
        ax[1].semilogy(phi, tau, linewidth=1.5, color='k', marker='o')
        #ax[1].semilogy(phi, vtau, linewidth=1.5, color='k', marker='o', linestyle='--')
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)
        ax[0].set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=17)
        ax[1].set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=17)
        ax[0].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
        ax[1].set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
        plt.tight_layout()
        #plt.savefig("/home/francesco/Pictures/dpm/diff-phi-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDynamicsVSactivity(dirName, figureName, plotDiff = True, save = False):
    Deff = []
    vDeff = []
    Teff = []
    tau = []
    vtau = []
    speed = []
    dataSetList = np.array(["2e-04", "4e-04", "6e-04", "8e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    #legendList = np.array(["$v_0=1.0 \\times 10^{-3}$", "$v_0=1.3 \\times 10^{-3}$", "$v_0=1.6 \\times 10^{-3}$", "$v_0=2.0 \\times 10^{-3}$", "$v_0=3.0 \\times 10^{-3}$"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0]+1)
    v0 = dataSetList.astype(float)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 9), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")):
            data = np.loadtxt(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics/corr-log.dat")
            timeStep = computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "dt")
            D0 = v0[i] * v0[i] / (2 * 1e-01)
            if(data[-1,2]<0.2):
                speed.append(v0[i])
                Teff.append(computeCorrelation.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "temperature"))
                #Deff.append(data[data[:,2]>np.exp(-1),1][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                #vDeff.append(data[data[:,2]>np.exp(-1),4][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                Deff.append(data[data[:,0]*timeStep>7e05,1][-1]/(4 * data[data[:,0]*timeStep>7e05,0][-1] * timeStep * D0))
                vDeff.append(data[data[:,0]*timeStep>7e05,4][-1]/(4 * data[data[:,0]*timeStep>7e05,0][-1] * timeStep * D0))
                tau.append(data[data[:,2]>np.exp(-1),0][-1]*timeStep)
                vtau.append(data[data[:,5]>np.exp(-1),0][-1]*timeStep)
                print("v0: " , v0[i], " Teff: ", Teff[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep, " timeStep:", timeStep, " maxStep:", data[-1,0])
            legendlabel = "$v_0=$" + dataSetList[i]
            #legendlabel = "$T_{eff}=$" + str(np.format_float_scientific(Teff[-1], 2))
            plotCorr(ax[0], data[1:,0]*timeStep, data[1:,1], data[1:,4], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = True)#, legendLabel = legendlabel)
            plotCorr(ax[1], data[1:,0]*timeStep, data[1:,2], data[1:,5], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
            #plotCorr(ax[1], data[1:,0]*timeStep, data[1:,3], data[1:,6], "$\\chi_4$", color = colorList(i/dataSetList.shape[0]))
    ax[1].plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax[1].legend(loc = "lower left", fontsize = 14)
    ax[1].set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=20)
    ax[0].set_ylim(8e-15,2)
    ax[1].set_xlim(4e-03, 4e08)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if(save=="save"):
        plt.savefig("/home/francesco/Pictures/dpm/corr-v0-" + figureName + ".png", transparent=True, format = "png")
    np.savetxt(dirName + "diff-tau-vs-activity.dat", np.column_stack((speed, Teff, Deff, tau, vDeff, vtau)))
    # plot diffusivity and relaxation time
    if(plotDiff == True):
        fig, ax = plt.subplots(1, 2, figsize = (13, 5), dpi = 120)
        #ax[0].plot(speed, Deff, linewidth=1.5, color='g', marker='o')
        ax[0].semilogx(Teff, vDeff, linewidth=1.5, color='k', marker='o')
        Teff = np.array(Teff)
        ax[1].loglog(1/Teff, tau, linewidth=1.5, color='k', marker='o')
        #ax[1].loglog(1/Teff, vtau, linewidth=1.5, color='k', marker='o', linestyle='--')
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)
        #ax[0].set_xlabel("$Propulsion$ $speed,$ $v_0$", fontsize=20)
        ax[0].set_xlabel("$Effective$ $temperature,$ $T_{eff}$", fontsize=20)
        ax[0].set_ylabel("$Diffusivisty,$ $D_{eff}$", fontsize=20)
        ax[1].set_xlabel("$Inverse$ $effective$ $temperature,$ $1/T_{eff}$", fontsize=20)
        ax[1].set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=20)
        #ax[0].set_ylim(0.00018,0.001009)
        #ax[1].set_ylim(6.7e03, 2.9e06)
        plt.tight_layout()
    if(save=="save"):
        plt.savefig("/home/francesco/Pictures/dpm/diff-v0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotTauVsTeff(dirName, figureName, sampleId):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    v0List = np.array(["2e-04", "4e-04", "6e-04", "8e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    v0ColorList = cm.get_cmap('plasma', dataSetList.shape[0])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(1, 2, figsize = (13, 5), dpi = 120)
    for i in range(dataSetList[:sampleId+1].shape[0]):
        data = np.loadtxt(dirName + os.sep + dataSetList[i] + "/ab/diff-tau-vs-activity.dat")
        phi = computeCorrelation.readFromParams(dirName + os.sep + dataSetList[i] + "/ab/Dr1e-01-v03e-03/dynamics/", "phi")
        legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi,4))
        ax[1].loglog(1/data[:,1], data[:,3], linewidth=1.2, color = colorList(i/dataSetList.shape[0]), marker='o', label = legendlabel)
        if(i == sampleId):
            for j in range(v0List.shape[0]):
                if(os.path.exists(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics/corr-log.dat")):
                    corr = np.loadtxt(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics/corr-log.dat")
                    timeStep = computeCorrelation.readFromParams(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics", "dt")
                    Teff = computeCorrelation.readFromParams(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics", "temperature")
                    legendlabel = "$T_{eff}=$" + str(np.format_float_scientific(Teff, 2))
                    plotCorr(ax[0], corr[1:,0]*timeStep, corr[1:,2], corr[1:,5], "$ISF(\\Delta t)$", color = v0ColorList(j/v0List.shape[0]), legendLabel = legendlabel)
    ax[0].plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.2, color=[0.5,0.5,0.5])
    ax[0].set_xlim(4e-03, 4e08)
    ax[0].legend(loc="lower left", fontsize=10)
    ax[1].legend(loc="lower right", fontsize=10)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=17)
    ax[0].set_ylabel("$ISF(\\Delta t)$", fontsize=17)
    ax[1].set_xlabel("$Inverse$ $effective$ $temperature,$ $1/T_{eff}$", fontsize=17)
    ax[1].set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax[1].set_ylim(2.8e03, 2.5e07)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/tau-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotTauVsPhi(dirName, figureName):
    dataSetList = np.array(["2e-04", "4e-04", "6e-04", "8e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(1, 2, figsize = (13, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        data = np.loadtxt(dirName + os.sep + "diff-tau-vs-phi-v0" + dataSetList[i] + ".dat")
        legendlabel = "$v_0=$" + dataSetList[i]
        ax[0].semilogy(data[:,0], data[:,2], linewidth=1.2, color = colorList(i/dataSetList.shape[0]), marker='o', label = legendlabel)
        ax[1].semilogy(data[:,0], data[:,3], linewidth=1.2, color = colorList(i/dataSetList.shape[0]), marker='o')
    ax[0].legend(loc="lower left", fontsize=11)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
    ax[1].set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[1].set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax[0].set_ylim(7.5e-06, 1.4e-03)
    ax[1].set_ylim(2.8e03, 2.5e07)
    ax[0].set_xlim(0.69,1.01)
    ax[1].set_xlim(0.69,1.01)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/tau-vs-phi-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def compareDynamics(fileName1, fileName2, figureName):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    fileNameList = np.array([fileName1, fileName2])
    colorList = ['r', 'b']
    for i in range(fileNameList.shape[0]):
        data = np.loadtxt(fileNameList[i] + ".dat")
        #msd plot
        plotCorr(ax[0], data[1:,0], data[1:,1], data[1:,4], "$MSD$", color = colorList[i], logy = True)
        # isf plot
        plotCorr(ax[1], data[1:,0], data[1:,2], data[1:,5], "$ISF$", color = colorList[i])
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/compare-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotVelocityCorrelation(dirName, figureName, numBins):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    bins, velcorr = computeCorrelation.computeVelocityHistogram(dirName, boxSize, nv, numBins)
    fig = plt.subplots(dpi = 150)
    ax = plt.gca()
    ax.plot(bins, velcorr, linewidth=1.2, color='k')
    ax.tick_params(axis="both", labelsize=17)
    ax.set_xlabel("$\\Delta r$", fontsize=20)
    ax.set_ylabel("$\\langle \\hat{v}_i \\cdot \\hat{v}_j \\rangle_{|\\vec{r}_i - \\vec{r}_j| = \\Delta r}$", fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/velcorr-" + figureName + ".png", transparent=False, format = "png")
    plt.show()


dirName = sys.argv[1]
whichPlot = sys.argv[2]

############################ check energy and force ############################
if(whichPlot == "energy"):
    plotEnergy(dirName)

elif(whichPlot == "energyscale"):
    figureName = sys.argv[3]
    plotEnergyScale(dirName, figureName)

elif(whichPlot == "compareenergy"):
    figureName = sys.argv[3]
    dirName1 = dirName + os.sep + sys.argv[4]
    dirName2 = dirName + os.sep + sys.argv[5]
    compareEnergy(dirName1, dirName2, figureName)

elif(whichPlot == "compareforce"):
    fileName = dirName + os.sep + sys.argv[3]
    dirName = dirName + os.sep + sys.argv[4]
    compareForces(dirName, fileName)

elif(whichPlot == "compareforcevideo"):
    figureName = sys.argv[3]
    fileName = dirName + os.sep + sys.argv[4]
    dirName = dirName + os.sep + sys.argv[5]
    compareForcesVideo(dirName, fileName, figureName)

########################### check and plot compression #########################
elif(whichPlot == "hop"):
    figureName = sys.argv[3]
    plotHOP(dirName, figureName)

elif(whichPlot == "hopphi"):
    figureName = sys.argv[3]
    plotHOPVSphi(dirName, figureName)

elif(whichPlot == "hexcomp"):
    figureName = sys.argv[3]
    plotHOPAlongCompression(dirName, figureName)

elif(whichPlot == "psi6p2"):
    figureName = sys.argv[3]
    numFrames = int(sys.argv[4])
    firstStep = float(sys.argv[5])
    stepFreq = float(sys.argv[6])
    plotPSI6P2(dirName, figureName, numFrames, firstStep, stepFreq)

elif(whichPlot == "velorder"):
    figureName = sys.argv[3]
    numFrames = int(sys.argv[4])
    firstStep = float(sys.argv[5])
    stepFreq = float(sys.argv[6])
    distTh = float(sys.argv[7])
    plotVelOrder(dirName, figureName, numFrames, firstStep, stepFreq, distTh)

elif(whichPlot == "velspace"):
    figureName = sys.argv[3]
    plotVelCorrSpace(dirName, figureName)

elif(whichPlot == "pcomp"):
    figureName = sys.argv[3]
    compute = sys.argv[4]
    plotParticleCompression(dirName, figureName, compute)

elif(whichPlot == "comp"):
    figureName = sys.argv[3]
    plotCompression(dirName, figureName)

elif(whichPlot == "comppsi6p2"):
    figureName = sys.argv[3]
    plotCompressionPSI6P2(dirName, figureName)

elif(whichPlot == "compset"):
    figureName = sys.argv[3]
    plotCompressionSet(dirName, figureName)

########################## plot correlation functions ##########################
elif(whichPlot == "check"):
    figureName = sys.argv[3]
    checkDynamics(dirName, figureName)

elif(whichPlot == "dynamics"):
    figureName = sys.argv[3]
    plotDynamics(dirName, figureName)

elif(whichPlot == "pscale"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleEnergyScale(dirName, sampleName, figureName)

elif(whichPlot == "pdensity"):
    sampleName = sys.argv[3]
    numBins = sys.argv[4]
    figureName = sys.argv[5]
    plotParticleDensity(dirName, sampleName, numBins, figureName)

elif(whichPlot == "pfdt"):
    figureName = sys.argv[3]
    Dr = float(sys.argv[4])
    driving = float(sys.argv[5])
    plotParticleFDT(dirName, figureName, Dr, driving)

elif(whichPlot == "pteff"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleTeff(dirName, sampleName, figureName)

elif(whichPlot == "pdyn"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleDynamics(dirName, sampleName, figureName)

elif(whichPlot == "pcomparefdt"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    compareParticleDynamicsFDT(dirName, sampleName, figureName)

elif(whichPlot == "pdynphi"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleDynamicsVSphi(dirName, sampleName, figureName)

elif(whichPlot == "pisfphi"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleISFVSphi(dirName, sampleName, figureName)

elif(whichPlot == "pdyntemp"):
    figureName = sys.argv[3]
    plotParticleDynamicsVStemp(dirName, figureName)

elif(whichPlot == "ptautemp"):
    figureName = sys.argv[3]
    plotParticleTauVStemp(dirName, figureName)

elif(whichPlot == "pdynq"):
    figureName = sys.argv[3]
    plotParticleDynamicsVSQ(dirName, figureName)

elif(whichPlot == "ptau"):
    sampleName = sys.argv[3]
    figureName = sys.argv[4]
    plotParticleTau(dirName, sampleName, figureName)

elif(whichPlot == "ptauactive"):
    figureName = sys.argv[3]
    plotParticleTauVsactivity(dirName, figureName)

elif(whichPlot == "vsactivity"):
    figureName = sys.argv[3]
    save = sys.argv[4]
    plotDynamicsVSactivity(dirName, figureName, save = save)

elif(whichPlot == "vsphi"):
    v0String = sys.argv[3]
    figureName = sys.argv[4]
    save = sys.argv[5]
    plotDynamicsVSphi(dirName, v0String, figureName, save = save)

elif(whichPlot == "tauvstemp"):
    figureName = sys.argv[3]
    sampleId = int(sys.argv[4])
    plotTauVsTeff(dirName, figureName, sampleId)

elif(whichPlot == "tauvsphi"):
    figureName = sys.argv[3]
    plotTauVsPhi(dirName, figureName)

elif(whichPlot == "compare"):
    figureName = sys.argv[3]
    fileName1 = dirName + os.sep + sys.argv[4]
    fileName2 = dirName + os.sep + sys.argv[5]
    compareDynamics(fileName1, fileName2, figureName)

elif(whichPlot == "velcorr"):
    figureName = sys.argv[3]
    numBins = int(sys.argv[4])
    plotVelocityCorrelation(dirName, figureName, numBins)

else:
    print("Please specify the type of plot you want")
