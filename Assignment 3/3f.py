import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SV:
    def __init__(self, dir):
        self.y = np.loadtxt(dir, dtype=np.float32, skiprows=1)
        self.n = len(self.y)

        self.x = np.linspace(0,10,self.n)

        self.mu = np.mean(self.y)

        # mean adjust the data again to correspond disturbances into N(0, 4.93)
        self.meaneps = -1.27
        self.vareps = 4.93

    def dataStats(self, label):
        #plt.figure()
        plt.plot(self.y, color="blue",label=label, linewidth=0.5)
        plt.legend(loc='upper right')
        plt.title('Pound/Dollar daily exchange rates')
        print('The mean  of the ' +label + ' is: ' + str(np.mean(self.y)))
        print('The stdev of the: ' +label + ' is: ' + str(np.std(self.y)))
        print('The variance of the: ' +label + ' is: ' + str(np.var(self.y)))
        plt.show()

    def meanAdjust(self):
        x = np.log((self.y - (self.mu))**2)
        self.y = x

    def QML(self, params):
        phi, omega, vareta = params

        v, f, k, a, P = self.KF(params)

        QML = -(self.n/2)*np.log(2*np.pi)

        for t in range(self.n):
            QML += -(1/2) * (np.log(f[t]) + (v[t]**2)/f[t])

        return -QML

    def KF(self, params):
        [phi, omega, vareta] = params

        # kalman filter
        v = np.zeros(self.n)
        f = np.zeros(self.n)
        k = np.zeros(self.n)
        a = np.zeros(self.n + 1)
        P = np.zeros(self.n + 1)

        a[0] = omega/(1 - phi)
        P[0] = vareta/(1 - phi**2)

        # Z = 1
        # H = vareps
        # T = phi
        # R = 1
        # Q = vareta

        for t in range(self.n):
            v[t] = self.y[t] - a[t] - self.meaneps
            f[t] = P[t] + self.vareps
            k[t] = phi * P[t]/f[t]

            a[t+1] = phi * a[t] + k[t]*v[t] + omega
            P[t+1] = phi**2 * P[t] + vareta - k[t]**2 * f[t]

        return  v, f, k, a, P

    def KS(self, v, f, k, a, P, phi):

        r = np.zeros(self.n)
        N = np.zeros(self.n)
        alphahat = np.zeros(self.n + 1)
        V = np.zeros(self.n + 1)

        r[self.n - 1] = 0
        N[self.n - 1] = 0

        for i in range(self.n - 1):
            t = self.n - 1 - i
            r[t - 1] = v[t]/f[t] + (phi - k[t]) * r[t]
            N[t - 1] = 1/f[t] + N[t] * (phi - k[t]) ** 2

        for i in range(self.n):
            t = self.n - i
            alphahat[t] = a[t] + P[t] * r[t-1]
            V[t] = P[t] - (P[t] ** 2) * N[t-1]

        return r, N, alphahat, V

    def FilteredPlot(self, a, P):
        plt.figure()
        plt.plot(self.x, a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='SV data', linewidth=1, alpha=0.3)

        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state ' + r'$\alpha_t$ ',fontsize=12)
        plt.draw()

    def SmoothedPlot(self, alphahat, V, title):
        plt.figure()
        plt.plot(self.x, alphahat[1:], color="blue", label='Smoothed state', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='SV data', linewidth=1, alpha=0.3)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title(title,fontsize=12)
        plt.draw()

    def pdot(self, ht, yt):
        pdot = -(1/2) + (1/2)*np.exp(-ht)*(yt - self.mu)**2

        return pdot

    def pdotdot(self, ht, yt):
        pdotdot = -(1/2)*np.exp(-ht)*(yt - self.mu)**2

        return pdotdot

    def ModeEstimation(self, params):
        phi, omega, vareta = params
        # first guess
        yminmu = self.y - self.mu

        guess = np.zeros(self.n)
        for t in range(self.n):
            guess[t] = np.log((1/self.n) * np.sum((yminmu)**2))

        score = np.sum(guess)

        while score > 0.1:
            A = np.zeros(self.n)
            z = np.zeros(self.n)

            for t in range(self.n):
                A[t] = -1/self.pdotdot(guess[t], self.y[t])
                z[t] = guess[t] + A[t] * self.pdot(guess[t], self.y[t])

            alphahat, V = self.modeKFS(z, A, [phi, omega, vareta])

            gmingcross = np.zeros(self.n)
            for t in range(self.n):
                gmingcross[t] = np.abs(guess[t] - alphahat[t])

            score = np.mean(gmingcross)
            print(score)

            guess = alphahat

        return guess, V, A

    def modeKFS(self, z, A, params):
        phi, omega, vareta = params
        # kalman filter
        v = np.zeros(self.n)
        f = np.zeros(self.n)
        k = np.zeros(self.n)
        a = np.zeros(self.n + 1)
        P = np.zeros(self.n + 1)

        a[0] = omega/(1 - phi)
        P[0] = vareta/(1 - phi**2)

        for t in range(self.n):
            v[t] = z[t] - a[t]
            f[t] = P[t] + A[t]
            k[t] = phi * P[t]/f[t]

            a[t+1] = phi * a[t] + k[t]*v[t] + omega
            P[t+1] = phi**2 * P[t] + vareta - k[t]**2 * f[t]

        r = np.zeros(self.n)
        N = np.zeros(self.n)
        alphahat = np.zeros(self.n + 1)
        V = np.zeros(self.n + 1)

        r[self.n - 1] = 0
        N[self.n - 1] = 0

        for i in range(self.n - 1):
            t = self.n - 1 - i
            r[t - 1] = v[t]/f[t] + (phi - k[t]) * r[t]
            N[t - 1] = 1/f[t] + N[t] * (phi - k[t]) ** 2

        for i in range(self.n):
            t = self.n - i
            alphahat[t] = a[t] + P[t] * r[t-1]
            V[t] = P[t] - (P[t] ** 2) * N[t-1]

        return alphahat, V

def main():
    # Read the data
    sv = SV('sv.dat')

    """a)"""
    #Data descriptions: plot/mean/var/etacross
    #sv.dataStats('returns')

    """b)"""
    sv.meanAdjust()
    #sv.dataStats('Log squared demeaned returns')

    """c)"""
    # phi, psi, vareta

    bnd = ((0.00001, 0.99999), (None, None), (0, None))
    [phi, omega, vareta] = minimize(sv.QML, (0.00001, 5, 5), bounds=bnd).x
    print("phi: {}".format(phi))
    print("Omega: {}".format(omega))
    print("Vareta: {}".format(vareta))

    """d)"""
    v, f, k, a, P = sv.KF([phi, omega, vareta])
    r, N, alphahat, V = sv.KS(v, f, k, a, P, phi)

    sv.FilteredPlot(a, P)
    sv.SmoothedPlot(alphahat, V, 'Smoothed estimate ' + r'$h_t$')

    """e)"""
    # Mode estimate
    guess, V, A = sv.ModeEstimation([phi, omega, vareta])

    sv.SmoothedPlot(guess, V, 'Smoothed mode of ' + r'$h_t$')

    # Estimate through importance sampling

    plt.show()

if __name__ == "__main__":
    main()
