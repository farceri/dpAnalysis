'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import itertools
import sys
import os
import utils
import dpCorrelation as dpCorr

def plotDPCorr(ax, x, y1, y2, ylabel, color, legendLabel = None, logx = True, logy = False):
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
    ax.set_ylabel("$Energy$", fontsize=15)
    #ax.legend(("$E_{pot}$", "$E_{tot}$"), fontsize=15, loc="lower right")
    ax.legend(("$E_{pot}$", "$E_{kin}$", "$E_{tot}$"), fontsize=15, loc="lower right")
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

def plotDPVelCorrTime(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, distTh = 0.1):
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    velcontact = []
    veldistance = []
    for i in stepList:
        stepVelcorr = dpCorr.computeVelCorrContact(dirName + os.sep + "t" + str(i), nv)
        velcontact.append(np.mean(stepVelcorr))
        stepVelcorr = dpCorr.computeVelCorrDistance(dirName + os.sep + "t" + str(i), boxSize, nv, distTh)
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

def plotDPVelCorrSpace(dirName, figureName):
    distThList = np.linspace(0.12,0.8,30)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    velcorr = []
    velcontact = np.mean(dpCorr.computeVelCorrContact(dirName + os.sep + "t100000", nv))
    for distTh in distThList:
        stepVelcorr = dpCorr.computeVelCorrDistance(dirName + os.sep + "t100000", boxSize, nv, distTh)
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


################################## plot dynamis ################################
def plotDPDynamics(fileName, figureName):
    data = np.loadtxt(fileName)
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    # msd plot
    plotDPCorr(ax[0], data[1:,0], data[1:,1], data[1:,4], "$MSD$", color = 'k', logy = True)
    # isf plot
    plotDPCorr(ax[1], data[1:,0], data[1:,2], data[1:,5], "$ISF$", color = 'k')
    # chi plot
    #plotDPCorr(ax[1], data[1:,0], data[1:,3], data[1:,6], "$\\chi_4$", color = 'k')
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    plt.savefig("/home/francesco/Pictures/" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def checkDPDynamics(dirName, figureName):
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
    plt.savefig("/home/francesco/Pictures/check-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def compareDPDynamics(fileName1, fileName2, figureName):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    fileNameList = np.array([fileName1, fileName2])
    colorList = ['r', 'b']
    for i in range(fileNameList.shape[0]):
        data = np.loadtxt(fileNameList[i] + ".dat")
        #msd plot
        plotDPCorr(ax[0], data[1:,0], data[1:,1], data[1:,4], "$MSD$", color = colorList[i], logy = True)
        # isf plot
        plotDPCorr(ax[1], data[1:,0], data[1:,2], data[1:,5], "$ISF$", color = colorList[i])
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/compare-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDPDynamicsVSphi(dirName, v0String, figureName, plotDiff = True, save = False):
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
            timeStep = utils.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "dt")
            currentPhi = utils.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "phi")
            data = np.loadtxt(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics/corr-log.dat")
            if(data[-1,2]<0.2):
                Teff.append(utils.readFromParams(dirName + dataSetList[i] + "/ab/Dr1e-01-v0" + v0String + "/dynamics", "temperature"))
                phi.append(currentPhi)
                Deff.append(data[data[:,2]>np.exp(-1),1][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                vDeff.append(data[data[:,2]>np.exp(-1),4][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                tau.append(data[data[:,2]>np.exp(-1),0][-1]*timeStep)
                vtau.append(data[data[:,5]>np.exp(-1),0][-1]*timeStep)
                print("phi: ", phi[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep, " timeStep:", timeStep)
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            plotDPCorr(ax[0], data[1:,0]*timeStep, data[1:,1], data[1:,4], "$MSD(\\Delta t)$", colorList(i/dataSetList.shape[0]), logy=True)
            plotDPCorr(ax[1], data[1:,0]*timeStep, data[1:,2], data[1:,5], "$ISF(\\Delta t)$", colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
            #plotDPCorr(ax[1], data[1:,0], data[1:,3], data[1:,6], "$\\chi_4$", colorList(i/dataSetList.shape[0]))
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

def plotDPDynamicsVSactivity(dirName, figureName, plotDiff = True, save = False):
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
            timeStep = utils.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "dt")
            D0 = v0[i] * v0[i] / (2 * 1e-01)
            if(data[-1,2]<0.2):
                speed.append(v0[i])
                Teff.append(utils.readFromParams(dirName + "/Dr1e-01-v0" + dataSetList[i] + "/dynamics", "temperature"))
                #Deff.append(data[data[:,2]>np.exp(-1),1][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                #vDeff.append(data[data[:,2]>np.exp(-1),4][-1]/(4 * data[data[:,2]>np.exp(-1),0][-1] * timeStep * D0))
                Deff.append(data[data[:,0]*timeStep>7e05,1][-1]/(4 * data[data[:,0]*timeStep>7e05,0][-1] * timeStep * D0))
                vDeff.append(data[data[:,0]*timeStep>7e05,4][-1]/(4 * data[data[:,0]*timeStep>7e05,0][-1] * timeStep * D0))
                tau.append(data[data[:,2]>np.exp(-1),0][-1]*timeStep)
                vtau.append(data[data[:,5]>np.exp(-1),0][-1]*timeStep)
                print("v0: " , v0[i], " Teff: ", Teff[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep, " timeStep:", timeStep, " maxStep:", data[-1,0])
            legendlabel = "$v_0=$" + dataSetList[i]
            #legendlabel = "$T_{eff}=$" + str(np.format_float_scientific(Teff[-1], 2))
            plotDPCorr(ax[0], data[1:,0]*timeStep, data[1:,1], data[1:,4], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = True)#, legendLabel = legendlabel)
            plotDPCorr(ax[1], data[1:,0]*timeStep, data[1:,2], data[1:,5], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), legendLabel = legendlabel)
            #plotDPCorr(ax[1], data[1:,0]*timeStep, data[1:,3], data[1:,6], "$\\chi_4$", color = colorList(i/dataSetList.shape[0]))
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

def plotDPTauVsTeff(dirName, figureName, sampleId):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    v0List = np.array(["2e-04", "4e-04", "6e-04", "8e-04", "1e-03", "1.3e-03", "1.6e-03", "2e-03", "3e-03"])
    v0ColorList = cm.get_cmap('plasma', dataSetList.shape[0])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(1, 2, figsize = (13, 5), dpi = 120)
    for i in range(dataSetList[:sampleId+1].shape[0]):
        data = np.loadtxt(dirName + os.sep + dataSetList[i] + "/ab/diff-tau-vs-activity.dat")
        phi = utils.readFromParams(dirName + os.sep + dataSetList[i] + "/ab/Dr1e-01-v03e-03/dynamics/", "phi")
        legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi,4))
        ax[1].loglog(1/data[:,1], data[:,3], linewidth=1.2, color = colorList(i/dataSetList.shape[0]), marker='o', label = legendlabel)
        if(i == sampleId):
            for j in range(v0List.shape[0]):
                if(os.path.exists(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics/corr-log.dat")):
                    corr = np.loadtxt(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics/corr-log.dat")
                    timeStep = utils.readFromParams(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics", "dt")
                    Teff = utils.readFromParams(dirName + str(sampleId) + "/ab/Dr1e-01-v0" + v0List[j] + "/dynamics", "temperature")
                    legendlabel = "$T_{eff}=$" + str(np.format_float_scientific(Teff, 2))
                    plotDPCorr(ax[0], corr[1:,0]*timeStep, corr[1:,2], corr[1:,5], "$ISF(\\Delta t)$", color = v0ColorList(j/v0List.shape[0]), legendLabel = legendlabel)
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

def plotDPTauVsPhi(dirName, figureName):
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

def plotDPVelCorrelation(dirName, figureName, numBins):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    bins, velcorr = utils.computeVelocityHistogram(dirName, boxSize, nv, numBins)
    fig = plt.subplots(dpi = 150)
    ax = plt.gca()
    ax.plot(bins, velcorr, linewidth=1.2, color='k')
    ax.tick_params(axis="both", labelsize=17)
    ax.set_xlabel("$\\Delta r$", fontsize=20)
    ax.set_ylabel("$\\langle \\hat{v}_i \\cdot \\hat{v}_j \\rangle_{|\\vec{r}_i - \\vec{r}_j| = \\Delta r}$", fontsize=20)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/dpm/velcorr-" + figureName + ".png", transparent=False, format = "png")
    plt.show()


if __name__ == '__main__':
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

    elif(whichPlot == "velcorrtime"):
        figureName = sys.argv[3]
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        distTh = float(sys.argv[7])
        plotDPVelCorrTime(dirName, figureName, numFrames, firstStep, stepFreq, distTh)

    elif(whichPlot == "velcorrspace"):
        figureName = sys.argv[3]
        plotDPVelCorrSpace(dirName, figureName)

########################## plot correlation functions ##########################
    elif(whichPlot == "dyn"):
        figureName = sys.argv[3]
        plotDPDynamics(dirName, figureName)

    elif(whichPlot == "check"):
        figureName = sys.argv[3]
        checkDPDynamics(dirName, figureName)

    elif(whichPlot == "compare"):
        figureName = sys.argv[3]
        fileName1 = dirName + os.sep + sys.argv[4]
        fileName2 = dirName + os.sep + sys.argv[5]
        compareDPDynamics(fileName1, fileName2, figureName)

    elif(whichPlot == "vsactivity"):
        figureName = sys.argv[3]
        save = sys.argv[4]
        plotDPDynamicsVSactivity(dirName, figureName, save = save)

    elif(whichPlot == "vsphi"):
        v0String = sys.argv[3]
        figureName = sys.argv[4]
        save = sys.argv[5]
        plotDPDynamicsVSphi(dirName, v0String, figureName, save = save)

    elif(whichPlot == "tauvstemp"):
        figureName = sys.argv[3]
        sampleId = int(sys.argv[4])
        plotDPTauVsTeff(dirName, figureName, sampleId)

    elif(whichPlot == "tauvsphi"):
        figureName = sys.argv[3]
        plotDPTauVsPhi(dirName, figureName)

    elif(whichPlot == "velcorr"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotDPVelCorrelation(dirName, figureName, numBins)

    else:
        print("Please specify the type of plot you want")
