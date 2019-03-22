"""
Plots figure 2.1 to 2.3
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import pylab
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from copy import copy

class Nile:
    def __init__(self):
        np.random.seed(1)

        self.data = [i.strip().split() for i in open("DK-data/Nile.dat").readlines()]
        self.data.pop(0)

        self.y = np.zeros(len(self.data))

        self.n = len(self.y)

        for i in range(self.n):
            self.y[i] = int(self.data[i][0])

        self.y = self.y.tolist()

        self.vareta = 1469.1
        self.vareps = 15099

        # initializations Kalman Filter
        self.T = float(input("Given the AR(1) + Noise model. Pick the value for phi in [-1, 1]. If LLM, phi = 1. "))

        self.a1 = 0

        if abs(self.T == 1):
            self.P1 = 10 ** 7
        elif abs(self.T < 1):
            self.P1 = self.vareta/(1 - self.T**2)

            meanY = np.mean(self.y)

            for i in range(self.n):
                self.y[i] = self.y[i] - meanY

        else:
            raise ValueError("The phi you've picked is not within the feasible range!")

        # initializations Kalman Smoother
        self.rn = 0
        self.Nn = 0

        # the label for the x-axis
        self.xYears = ["", 1880, "", 1900, "", 1920, "", 1940, "", 1960]
        self.x = np.linspace(0,10,self.n)

    def Ldc(self, q):
        v = [0 for t in range(self.n)]
        fstar = [0 for t in range(self.n)]
        k = [0 for t in range(self.n)]
        a = [0 for t in range(self.n + 1)]
        Pstar = [0 for t in range(self.n + 1)]

        # diffuse prior density
        a[0] = 0
        Pstar[0] = 10**7

        a[1] = self.y[0]
        Pstar[1] = 1 + q

        #Kalman filter
        for t in range(1, self.n):
            v[t] = self.y[t] - a[t]
            fstar[t] = Pstar[t] + 1
            a[t+1] = a[t] + Pstar[t]/fstar[t]*v[t]
            Pstar[t+1] = Pstar[t]*(1 - Pstar[t]/fstar[t]) + q

        varepshat = 0
        for t in range(1, self.n):
            varepshat += (v[t]**2) / fstar[t]

        varepshat = varepshat/(self.n-1)

        Ldc = -(self.n/2)*np.log(2*np.pi) - (self.n - 1)/2 - ((self.n - 1)/2)*np.log(varepshat)

        for t in range(1, self.n):
            Ldc += -(1/2)*np.log(fstar[t])

        return Ldc

    def varepshat(self, q):
        v = [0 for t in range(self.n)]
        fstar = [0 for t in range(self.n)]
        k = [0 for t in range(self.n)]
        a = [0 for t in range(self.n + 1)]
        Pstar = [0 for t in range(self.n + 1)]

        # diffuse prior density
        a[0] = 0
        Pstar[0] = 10**7

        a[1] = self.y[0]
        Pstar[1] = 1 + q

        #Kalman filter
        for t in range(1, self.n):
            v[t] = self.y[t] - a[t]
            fstar[t] = Pstar[t] + 1
            a[t+1] = a[t] + Pstar[t]/fstar[t]*v[t]
            Pstar[t+1] = Pstar[t]*(1 - Pstar[t]/fstar[t]) + q

        varepshat = 0
        for t in range(1, self.n):
            varepshat += (v[t]**2) / fstar[t]

        varepshat = varepshat/(self.n-1)

        return varepshat

    def MLEar1plusnoise(self, params):
        phi, vareps, vareta = params
        v = np.zeros(self.n)
        f = np.zeros(self.n)
        k = np.zeros(self.n)
        a = np.zeros(self.n + 1)
        P = np.zeros(self.n + 1)

        a[0] = 0
        P[0] = vareta/(1 - phi**2)

        for t in range(self.n):
            v[t] = self.y[t] - a[t]
            f[t] = P[t] + vareps
            k[t] = phi * P[t]/f[t]

            a[t+1] = phi * a[t] + k[t]*v[t]
            P[t+1] = phi**2 * P[t] + vareta - k[t]**2 * f[t]

        MLE = -(self.n/2)*np.log(2*np.pi)

        for t in range(self.n):
            MLE += -(1/2) * (np.log(f[t]) + (v[t]**2)/f[t])

        return -MLE

    def KalmanFilter(self, y):
        self.v = np.zeros(self.n)
        self.f = np.zeros(self.n)
        self.k = np.zeros(self.n)

        self.a = np.zeros(self.n + 1)
        self.a[0] = self.a1
        self.P = np.zeros(self.n + 1)
        self.P[0] = self.P1

        for t in range(self.n):
            if y[t] == None :
                # doesn't really matter
                self.v[t] = 0
                self.f[t] = 10 ** 7
                self.k[t] = 0
            else:
                self.v[t] = y[t] - self.a[t]
                self.f[t] = self.P[t] + self.vareps
                self.k[t] = self.T * self.P[t]/self.f[t]

            self.a[t+1] = self.T*self.a[t] + self.k[t]*self.v[t]
            self.P[t+1] = (self.T**2) * self.P[t] + self.vareta - (self.k[t]**2) * self.f[t]

    def KalmanSmoother(self):
        self.r = np.zeros(self.n)
        self.N = np.zeros(self.n)
        self.alphahat = np.zeros(self.n + 1)
        self.V = np.zeros(self.n + 1)

        self.r[self.n - 1] = self.rn
        self.N[self.n - 1] = self.Nn
        for i in range(self.n - 1):
            t = self.n - 1 - i
            self.r[t - 1] = self.v[t]/self.f[t] + (self.T - self.k[t]) * self.r[t]
            self.N[t - 1] = 1/self.f[t] + self.N[t] * (self.T - self.k[t]) ** 2

        for i in range(self.n):
            t = self.n - i
            self.alphahat[t] = self.a[t] + self.P[t] * self.r[t-1]
            self.V[t] = self.P[t] - (self.P[t] ** 2) * self.N[t-1]

    def DisturbanceSmoother(self):
        self.mu = np.zeros(self.n)
        self.epshat = np.zeros(self.n)
        self.stdepshatgiveny = np.zeros(self.n)
        self.D = np.zeros(self.n)
        self.etahat = np.zeros(self.n)
        self.stdetagiveny = np.zeros(self.n)

        for t in range(self.n):
            self.mu[t] = self.v[t]/self.f[t] - self.k[t] * self.r[t]
            self.epshat[t] = self.vareps * self.mu[t]

            self.D[t] = 1/self.f[t] + (self.k[t]**2) * self.N[t]
            self.stdepshatgiveny[t] = np.sqrt(self.vareps - (self.vareps**2) * self.D[t])

            self.etahat[t] = self.vareta * self.r[t]
            self.stdetagiveny[t] = np.sqrt(self.vareta - (self.vareta**2) * self.N[t])

    def fig1(self):
        """ 2.1.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.a[i+1] - 1.645*np.sqrt(self.P[i+1])
            upperboundCI[i] = self.a[i+1] + 1.645*np.sqrt(self.P[i+1])

        plt.figure()
        plt.plot(self.x, self.a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(self.x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(self.x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state ' + r'$\alpha_t$ ' + 'and its 90% confidence intervals',fontsize=12)
        plt.draw()

        """ 2.1.ii """
        plt.figure()
        plt.plot(self.x, self.P[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state variance ' + r'$P_t$',fontsize=12)
        plt.draw()

        """ 2.1.iii """
        self.v[0] = 0
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.v, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction errors ' + r'$v_t$',fontsize=12)
        plt.draw()

        """ 2.1.iv """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.ylim(bottom=20000, top=32500)
        plt.plot(self.x, self.f, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction variance ' + r'$F_t$',fontsize=12)
        plt.draw()

    def fig2(self):
        """ 2.2.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.alphahat[i+1] - 1.645*np.sqrt(self.V[i+1])
            upperboundCI[i] = self.alphahat[i+1] + 1.645*np.sqrt(self.V[i+1])

        plt.figure()
        plt.plot(self.x, self.alphahat[1:], color="blue", label='Smoothed state', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(self.x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(self.x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Nile data and output of state smoothing recursion',fontsize=12)
        plt.draw()

        """ 2.2.ii """
        plt.figure()
        plt.ylim(bottom=2000, top=4200)
        plt.plot(self.x, self.V[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state variance ' + r'$V_t$',fontsize=12)
        plt.draw()

        """ 2.2.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.r, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing cumulant ' + r'$r_t$',fontsize=12)
        plt.draw()

        """ 2.2.iv """
        plt.figure()
        plt.plot(self.x, self.N, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing variance cumulant ' + r'$N_t$',fontsize=12)
        plt.draw()

    def fig3(self):
        """ 2.3.i """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.epshat, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation error ' + r'$\hat{\epsilon}_t$',fontsize=12)
        plt.draw()

        """ 2.3.ii """
        plt.figure()
        plt.plot(self.x, self.stdepshatgiveny, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation error standard error ' + r'$\sqrt{Var(\epsilon_t | y)}$',fontsize=12)
        plt.draw()

        """ 2.3.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.etahat, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State error ' + r'$\hat{\eta}_t$',fontsize=12)
        plt.draw()

        """ 2.3.iv """
        plt.figure()
        plt.plot(self.x, self.stdetagiveny, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State error standard error ' + r'$\sqrt{Var(\eta_t | y)}$',fontsize=12)
        plt.draw()

    def simulation(self):
        epscross = np.random.normal(0, np.sqrt(self.vareps), self.n)
        etacross = np.random.normal(0, np.sqrt(self.vareta), self.n)

        self.alphacross = np.zeros(self.n + 1)

        if self.T == 1:
            self.alphacross[0] = self.a[1]
        else:
            self.alphacross[0] = np.random.normal(0, np.sqrt(self.P1))

        self.ycross = np.zeros(self.n)

        for t in range(self.n):
            self.ycross[t] = self.alphacross[t] + epscross[t]
            self.alphacross[t + 1] = self.T * self.alphacross[t] + etacross[t]

    def KalmanFilterSim(self, y):
        self.vsim = np.zeros(self.n)
        self.fsim = np.zeros(self.n)
        self.ksim = np.zeros(self.n)

        self.asim = np.zeros(self.n + 1)
        self.asim[0] = self.a1
        self.Psim = np.zeros(self.n + 1)
        self.Psim[0] = self.P1

        for t in range(self.n):
            if y[t] == None :
                # doesn't really matter
                self.vsim[t] = 0
                self.fsim[t] = 10 ** 7
                self.ksim[t] = 0
            else:
                self.vsim[t] = y[t] - self.asim[t]
                self.fsim[t] = self.Psim[t] + self.vareps
                self.ksim[t] = self.T * self.Psim[t]/self.fsim[t]

            self.asim[t+1] = self.T*self.asim[t] + self.ksim[t]*self.vsim[t]
            self.Psim[t+1] = (self.T**2) * self.Psim[t] + self.vareta - (self.ksim[t]**2) * self.fsim[t]

    def KalmanSmootherSim(self):
        self.rsim = np.zeros(self.n)
        self.Nsim = np.zeros(self.n)
        self.alphahatcross = np.zeros(self.n + 1)
        self.Vsim = np.zeros(self.n + 1)

        self.rsim[self.n - 1] = self.rn
        self.Nsim[self.n - 1] = self.Nn
        for i in range(self.n - 1):
            t = self.n - 1 - i
            self.rsim[t - 1] = self.vsim[t]/self.fsim[t] + (self.T - self.ksim[t]) * self.rsim[t]
            self.Nsim[t - 1] = 1/self.fsim[t] + self.Nsim[t] * (self.T - self.ksim[t]) ** 2

        for i in range(self.n):
            t = self.n - i
            self.alphahatcross[t] = self.asim[t] + self.Psim[t] * self.rsim[t-1]
            self.Vsim[t] = self.Psim[t] - (self.Psim[t] ** 2) * self.Nsim[t-1]

    def alphatilde(self):
        self.alphatilde = np.zeros(self.n + 1)

        for t in range(self.n):
            self.alphatilde[t] = self.alphahat[t] + self.alphacross[t] - self.alphahatcross[t]

    def epstilde(self):
        self.epstilde = np.zeros(self.n)

        for t in range(self.n):
            self.epstilde[t] = self.y[t] - self.alphatilde[t]

    def etatilde(self):
        self.etatilde = np.zeros(self.n)

        for t in range(self.n):
            self.etatilde[t] = self.alphatilde[t+1] - self.alphatilde[t]

    def fig4(self):
        """ 2.4.i """
        plt.figure()
        plt.plot(self.x, self.alphahat[1:], label=r'$\hat{\alpha}_t$',color="blue", linewidth=0.5)
        plt.scatter(self.x, self.alphacross[1:], label=r'$\alpha_{t}^{+}$', color="red", s=10)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state ' + r'$\hat{\alpha}_t$ ' + 'and sample ' + r'$\alpha_{t}^{+}$',fontsize=12)
        plt.draw()

        """ 2.4.ii """
        plt.figure()
        plt.plot(self.x, self.alphahat[1:], label=r'$\hat{\alpha}_t$',color="blue", linewidth=0.5)
        plt.scatter(self.x, self.alphatilde[:-1], label=r'$\widetilde{\alpha}_{t}^{+}$', color="red", s=10)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state ' + r'$\hat{\alpha}_t$ ' + 'and sample ' + r'$\widetilde{\alpha}_{t}^{+}$',fontsize=12)
        plt.draw()

        """ 2.4.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.epshat, label=r'$\hat{\epsilon}_t$', color="blue", linewidth=0.5)
        plt.scatter(self.x, self.epstilde, label= r'$\widetilde{\epsilon}_t$', color="red", s=10)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed observation error ' + r'$\hat{\epsilon}_t$ ' + 'and sample ' + r'$\widetilde{\epsilon}_t$',fontsize=12)
        plt.draw()

        """ 2.4.iv """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.ylim(bottom=-400, top=200)
        plt.plot(self.x, self.etahat, label=r'$\hat{\eta}_t$', color="blue", linewidth=0.5)
        plt.scatter(self.x, self.etatilde, label= r'$\widetilde{\eta}_t$', color="red", s=10)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state error ' + r'$\hat{\eta}_t$ ' + 'and sample ' + r'$\widetilde{\eta}_t$',fontsize=12)
        plt.draw()


    def treatAsMissing(self):
        for i in range(20, 40):
            self.y[i] = int(self.y[i])
            self.y[i] = None
        for i in range(60, 80):
            self.y[i] = int(self.y[i])
            self.y[i] = None

    def fig5(self):
        """ 2.5.i """
        plt.figure()
        plt.plot(self.x, self.a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state ' + r'$\alpha_t$ ' + '(extrapolation)',fontsize=12)
        plt.draw()

        """ 2.5.ii """
        plt.figure()
        plt.plot(self.x, self.P[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state variance ' + r'$P_t$',fontsize=12)
        plt.draw()

        """ 2.5.iii """
        plt.figure()
        plt.plot(self.x, self.alphahat[1:], color="blue", label='Smoothed state', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state ' + r'$\hat{\alpha_t}$ ' + '(interpolation)',fontsize=12)
        plt.draw()

        """ 2.5.iv """
        plt.figure()
        if self.T == 1:
            plt.ylim(bottom=2000, top=10000)
        plt.plot(self.x, self.V[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state variance ' + r'$V_t$',fontsize=12)
        plt.draw()

    def resetY(self):
        self.data = [i.strip().split() for i in open("DK-data/Nile.dat").readlines()]
        self.data.pop(0)

        self.y = np.zeros(len(self.data))

        self.n = len(self.y)

        for i in range(self.n):
            self.y[i] = int(self.data[i][0])

        if abs(self.T == 1):
            self.P1 = 10 ** 7
        elif abs(self.T < 1):
            self.P1 = self.vareta/(1 - self.T**2)

            meanY = np.mean(self.y)

            for i in range(self.n):
                self.y[i] = self.y[i] - meanY

        self.y = self.y.tolist()

    def extendData(self):
        for i in range(30):
            self.y.append(None)

        self.n = len(self.y)

        # relabel for the x-axis
        self.xYears = [1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
        self.x = np.linspace(0,14,self.n)


    def fig6(self):
        """ 2.6.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.a[i+1] - 0.67449*np.sqrt(self.P[i+1])
            upperboundCI[i] = self.a[i+1] + 0.67449*np.sqrt(self.P[i+1])

        plt.figure()
        plt.plot(self.x, self.a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(self.x, lowerboundCI, color="red", label='50% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(self.x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(14), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State forecast ' + r'$\alpha_t$ ' + 'and its 50% confidence intervals',fontsize=12)
        plt.draw()

        """ 2.6.ii """
        plt.figure()
        plt.plot(self.x, self.P[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(14), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State variance ' + r'$P_t$',fontsize=12)
        plt.draw()

        """ 2.6.iii """
        plt.figure()
        plt.plot(self.x, self.a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.xticks(np.arange(14), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation forecast ' + r'$E(y_t | Y_{t-1})$ ',fontsize=12)
        plt.draw()

        """ 2.6.iv """
        self.f = self.P[1:] + self.vareps
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.f, color="blue", linewidth=0.5)
        plt.xticks(np.arange(14), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation forecast variance ' + r'$F_t$',fontsize=12)
        plt.draw()

    def standardResidual(self):
        self.e = np.zeros(self.n)

        for t in range(self.n):
            self.e[t] = self.v[t]/np.sqrt(self.f[t])

    def fig7(self):
        """ 2.7.i """
        plt.figure()
        plt.plot(self.x, self.e, color="blue", linewidth=0.5)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.xticks(np.arange(14), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Standardised residual ' + r'$e_t$',fontsize=12)
        plt.draw()

        """ 2.7.ii """
        plt.figure()
        plt.hist(self.e, 13, density=True, rwidth=0.85)
        density = gaussian_kde(self.e)
        xs = np.linspace(-4,4,100)
        plt.plot(xs,density(xs))
        plt.title('Histogram and estimated density for ' + r'$e_t$',fontsize=12)
        plt.draw()

        """ 2.7.iii """
        sm.qqplot(self.e, line='45')
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.title('QQ-plot',fontsize=12)
        plt.draw()

        """ 2.7.iv """
        statsmodels.graphics.tsaplots.plot_acf(self.e, lags=17)
        plt.draw()

    def mustar(self):
        self.mustar = np.zeros(self.n)
        for t in range(self.n):
            self.mustar[t] = self.mu[t]/np.sqrt(self.D[t])

    def rstar(self):
        self.rstar = np.zeros(self.n)
        for t in range(self.n):
            if self.N[t] != 0:
                self.rstar[t] = self.r[t]/np.sqrt(self.N[t])

    def fig8(self):
        """ 2.8.i """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axhline(y=2, color='k', linewidth=0.5, linestyle="--")
        plt.axhline(y=-2, color='k', linewidth=0.5, linestyle="--")
        plt.plot(self.x, self.mustar, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation residual ' + r'$\mu_{t}^{*}$',fontsize=12)
        plt.draw()

        """ 2.8.ii """
        plt.figure()
        plt.hist(self.mustar, 13, density=True, rwidth=0.85)
        density = gaussian_kde(self.mustar)
        xs = np.linspace(-4,4,100)
        plt.plot(xs,density(xs))
        plt.title('Histogram and estimated density for ' + r'$\mu_{t}^{*}$',fontsize=12)
        plt.draw()

        """ 2.8.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axhline(y=2, color='k', linewidth=0.5, linestyle="--")
        plt.axhline(y=-2, color='k', linewidth=0.5, linestyle="--")
        plt.plot(self.x, self.rstar, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State residual ' + r'$r_{t}^{*}$',fontsize=12)
        plt.draw()

        """ 2.8.ii """
        plt.figure()
        plt.hist(self.rstar[:-1], 13, density=True, rwidth=0.85)
        density = gaussian_kde(self.rstar[:-1])
        xs = np.linspace(-4,4,100)
        plt.plot(xs,density(xs))
        plt.title('Histogram and estimated density for ' + r'$r_{t}^{*}$',fontsize=12)
        plt.draw()

def main():
    nile = Nile()

    """ Compute parameter estimates """

    if nile.T == 1:
        bnd = ((1e-6, None),)
        qhat = minimize(lambda q: -nile.Ldc(q), 1e-6, bounds=bnd).x[0]
        nile.vareps = nile.varepshat(qhat)
        nile.vareta = qhat*nile.vareps

        print("Estimated through MLE")
        print("vareps: {}, vareta: {}".format(nile.vareps, nile.vareta))
    else:
        bnd = ((-0.99999, 0.99999),(1e-6, None), (1e-6, None))
        estimates = minimize(nile.MLEar1plusnoise, (0, 1e-6, 1e-6), bounds=bnd).x

        [nile.T, nile.vareps, nile.vareta] = estimates

        print("Estimated through MLE")
        print("phi: {}, vareps: {}, vareta: {}".format(nile.T, nile.vareps, nile.vareta))
        
    nile.KalmanFilter(nile.y)
    #nile.fig1()

    nile.KalmanSmoother()
    #nile.fig2()

    nile.DisturbanceSmoother()
    #nile.fig3()

    nile.simulation()
    nile.KalmanFilterSim(nile.ycross)
    nile.KalmanSmootherSim()
    nile.alphatilde()
    nile.epstilde()
    nile.etatilde()
    nile.fig4()

    nile.treatAsMissing()
    nile.KalmanFilter(nile.y)
    nile.KalmanSmoother()
    #nile.fig5()

    nile.resetY()
    nile.extendData()
    nile.KalmanFilter(nile.y)
    #nile.fig6()

    nile = Nile()
    nile.KalmanFilter(nile.y)
    nile.standardResidual()
    #nile.fig7()

    nile.KalmanSmoother()
    nile.DisturbanceSmoother()
    nile.mustar()
    nile.rstar()
    #nile.fig8()

    plt.show()

if __name__ == "__main__":
    main()
