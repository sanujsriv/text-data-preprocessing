import mpmath
import seaborn
import numpy
import matplotlib.pyplot as plt


def cdf(x):
    """
    Cumulative distribution function (CDF) of the raised cosine distribution.
    The CDF of the raised cosine distribution is
        F(x) = (pi + x + sin(x))/(2*pi)
    """
    with mpmath.extradps(5):
        x = mpmath.mpf(x)
        if x <= -mpmath.pi:
            return mpmath.mp.zero
        if x >= mpmath.pi:
            return mpmath.mp.one
        return mpmath.mpf('1/2') + (x + mpmath.sin(x))/(2*mpmath.pi)

def _p2(t):
    t = mpmath.mpf(t)
    return (0.5
            + t*(-0.06532856457583547
                 + t*(0.0020893844847965047
                      + t*-1.0233693819385904e-05)))


def _q2(t):
    t = mpmath.mpf(t)
    return (1.0
            + t*(-0.15149046248500425
                 + t*(0.006293153604697265
                      + t*-6.042645518776793e-05)))


def invcdf(p):
    """
    Inverse of the CDF of the raised cosine distribution.
    """
    with mpmath.extradps(5):
        p = mpmath.mpf(p)

        if p < 0 or p > 1:
            return mpmath.nan
        if p == 0:
            return -mpmath.pi
        if p == 1:
            return mpmath.pi

        y = mpmath.pi*(2*p - 1)
        y2 = y**2
        x = y * _p2(y2) / _q2(y2)

        solver = 'mnewton'
        x = mpmath.findroot(f=lambda t: cdf(t) - p,
                            x0=x,
                            df=lambda t: (1 + mpmath.cos(t))/(2*mpmath.pi),
                            df2=lambda t: -mpmath.sin(t)/(2*mpmath.pi),
                            solver=solver)

        return x


def RaiseCosineRNG():

    u = 5
    s = 0.01
    s = s/3.14

    z = numpy.random.uniform(low=0, high=1, size=10000)

    p = []
    for i in range(10000):
        p.append((float(invcdf(z[i]))*s)+u)

    f = plt.figure()
    seaborn.distplot(p)
    plt.show()

RaiseCosineRNG()
