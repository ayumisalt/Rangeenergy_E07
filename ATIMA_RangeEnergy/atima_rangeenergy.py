import numpy as np
import os
import pycatima

ATIMA_splines = None
ATIMA_Z_max = 8

emulsion = pycatima.Material()
emulsion.add_element(1.00784, 1, 0.0161)
emulsion.add_element(12.011, 6, 0.0922)
emulsion.add_element(14.0067, 7, 0.0307)
emulsion.add_element(15.999, 8, 0.0847)
emulsion.add_element(32.065, 16, 0.0003)
emulsion.add_element(79.904, 35, 0.3229)
emulsion.add_element(107.8682, 47, 0.444)
emulsion.add_element(126.90447, 53, 0.0094)

def GetEmulsionSpline(EmulsionType, SplineDir="./splines_emul_1.41"):
    Zs = None
    MassRatios = None
    if EmulsionType == "E07_standard":
        TargetZs = [1, 6, 7, 8, 16, 47, 35, 53]
        MassRatios = [1.61, 9.22, 3.07, 8.47, 0.03, 44.40, 32.29, 0.94]
        # MassRatios = [1.5, 9.3, 3.1, 6.8, 0.2, 45.40, 33.4, 0.3]

    else:
        raise
    return GetSpline(SplineDir, TargetZs, MassRatios)


def GetSpline(SplineDir, TargetZs, MassRatios):
    MassRatioSum = np.sum(MassRatios)
    NormMassRatios = [r / MassRatioSum for r in MassRatios]
    BeamZs = [1, 2, 3, 4, 5, 6, 7, 8]
    MassUs = [
        1.007825017,
        4.002603054,
        7.016004086,
        9.012182236,
        10.01293659,
        12,
        14.00307369,
        15.99491501,
    ]
    splines = {}
    for BeamZ, MassU in zip(BeamZs, MassUs):
        splines[BeamZ] = {}
        spline = {}
        Energy = None
        spline["dEdx"] = []
        spline["Range"] = []
        spline["RangeStraggling"] = []
        for TargetZ in TargetZs:
            filename = f"BeamZ{BeamZ}_TargetZ{TargetZ}.txt"
            filepath = os.path.join(os.path.dirname(__file__), SplineDir, filename)
            if not os.path.exists(filepath):
                print(filepath, "is not found.")
                raise
            sp = np.loadtxt(filepath)
            Energy = sp.T[0]
            spline["dEdx"].append(sp.T[1])
            spline["Range"].append(sp.T[2])
            spline["RangeStraggling"].append(sp.T[3])
        BetaGamma = GetBetaGamma(Energy, MassU)
        CompositedEdx = GetCompositedEdx(spline["dEdx"], NormMassRatios)
        CompositeRange = GetCompositedRange(spline["Range"], NormMassRatios)
        CompositeRangeStraggling = GetCompositedRangeStraggling(
            spline["Range"], spline["RangeStraggling"], CompositeRange, NormMassRatios
        )

        splines[BeamZ]["BetaGamma"] = BetaGamma
        splines[BeamZ]["dEdx"] = CompositedEdx
        splines[BeamZ]["Range"] = CompositeRange
        splines[BeamZ]["RangeStraggling"] = CompositeRangeStraggling
    return splines


def GetCompositedEdx(dEdx, NormMassRatios):
    CompositedEdx = []
    for i in range(len(dEdx[0])):
        x = 0
        for j in range(len(NormMassRatios)):
            x += dEdx[j][i] * NormMassRatios[j]
        CompositedEdx.append(x)
    return CompositedEdx


def GetCompositedRange(Range, NormMassRatios):
    CompositeRange = []
    for i in range(len(Range[0])):
        x = 0
        for j in range(len(NormMassRatios)):
            if Range[j][i] != 0:
                x += NormMassRatios[j] / Range[j][i]
        if x != 0:
            CompositeRange.append(1.0 / x)
        else:
            CompositeRange.append(0)
    return CompositeRange


def GetCompositedRangeStraggling(
    Range, RangeStraggling, CompositeRange, NormMassRatios
):
    CompositeRangeStraggling = []
    for i in range(len(Range[0])):
        x = 0
        for j in range(len(NormMassRatios)):
            if Range[j][i] != 0:
                x += NormMassRatios[j] / (RangeStraggling[j][i] / Range[j][i])
        if x != 0:
            CompositeRangeStraggling.append(CompositeRange[i] / x)
        else:
            CompositeRangeStraggling.append(0)
    return CompositeRangeStraggling


amu = 931.49410


def GetBetaGamma(Energy, MassU):
    BetaGamma = []
    for x in Energy:
        KE = x * MassU
        M = MassU * amu
        Mom = ((M + KE) ** 2 - M**2) ** 0.5
        BetaGamma.append(Mom / M)
    return BetaGamma


def BetaGamma2KE(Mass, BetaGamma):
    Mom = BetaGamma * Mass
    return (Mass**2 + Mom**2) ** 0.5 - Mass


def KE2BetaGamma(Mass, KE):
    Mom = ((Mass + KE) ** 2 - Mass**2) ** 0.5
    return Mom / Mass


from scipy import interpolate
from scipy import integrate


def RangeStragglingFromRange(Mass, Range, Z, density):
    """
    To get RangeStraggling from Range

    Parameters
    ----------
        Mass    : float MeV/c2
        Range   : float um
        Z       : int Charge. Allow negative values.
        density : float Emulsion density g/cm3
    Returns
    ----------
        RangeStraggling : float std. dev. um
    """

    ke = KEfromRange(Mass, Range, Z, density)
    projectile = pycatima.Projectile(Mass/amu, abs(Z))
    results = pycatima.calculate(projectile(ke /(Mass/amu)), emulsion)
    result = results.sigma_r * 10000 / density

    return result


def RangeStragglingFromKE(Mass, KE, Z, density):
    """
    To get RangeStraggling from KE

    Parameters
    ----------
        Mass    : float MeV/c2
        KE      : float MeV
        Z       : int Charge. Allow negative values.
        density : float Emulsion density g/cm3
    Returns
    ----------
        RangeStraggling : float std. dev. um
    """

    projectile = pycatima.Projectile(Mass/amu, abs(Z))
    results = pycatima.calculate(projectile(KE /(Mass/amu)), emulsion)
    result = results.sigma_r * 10000 / density
    
    return result


def KEfromRange(Mass, Range, Z, density):
    """
    To get KE from Range

    Parameters
    ----------
        Mass    : float MeV/c2
        Range   : float um
        Z       : int Charge. Allow negative values.
        density : float Emulsion density g/cm3
    Returns
    ----------
        KE      : float MeV
    """
    if abs(Z) > ATIMA_Z_max:
        print(f"abs(Z) > {ATIMA_Z_max} is not supported.")
        raise
    if ATIMA_splines == None:
        print("ATIMA_splines is not given.")
        raise
    Energyies = []
    InvdEdxs = []
    for bg, dEdx in zip(
        ATIMA_splines[abs(Z)]["BetaGamma"], ATIMA_splines[abs(Z)]["dEdx"]
    ):
        Energyies.append(BetaGamma2KE(Mass, bg))
        InvdEdxs.append(1.0 / dEdx)
    Ranges = integrate.cumulative_trapezoid(InvdEdxs, Energyies, initial=0)
    f = interpolate.interp1d(Ranges, Energyies, kind="cubic")
    return f(Range * density / 10.0 / 1000)


def RangeFromKE(Mass, KE, Z, density):
    """
    To get Range from KE

    Parameters
    ----------
        Mass    : float MeV/c2
        KE      : float MeV
        Z       : int Charge. Allow negative values.
        density : float Emulsion density g/cm3
    Returns
    ----------
        Range   : float um
    """
    if abs(Z) > ATIMA_Z_max:
        print(f"abs(Z) > {ATIMA_Z_max} is not supported.")
        raise
    if ATIMA_splines == None:
        print("ATIMA_splines is not given.")
        raise
    Energyies = []
    InvdEdxs = []
    for bg, dEdx in zip(
        ATIMA_splines[abs(Z)]["BetaGamma"], ATIMA_splines[abs(Z)]["dEdx"]
    ):
        Energyies.append(BetaGamma2KE(Mass, bg))
        InvdEdxs.append(1.0 / dEdx)
    Ranges = integrate.cumulative_trapezoid(InvdEdxs, Energyies, initial=0)
    f = interpolate.interp1d(Energyies, Ranges, kind="cubic")
    return f(KE) / density * 10.0 * 1000


def DensityFromKERange(Mass, KE, Range, Z):
    """
    To get density from KE and Range

    Parameters
    ----------
        Mass    : float MeV/c2
        KE      : float MeV
        Range   : float um
        Z       : int Charge. Allow negative values.
    Returns
    ----------
        density : float Emulsion density g/cm3
    """
    if abs(Z) > ATIMA_Z_max:
        print(f"abs(Z) > {ATIMA_Z_max} is not supported.")
        raise
    if ATIMA_splines == None:
        print("ATIMA_splines is not given.")
        raise
    return RangeFromKE(Mass, KE, Z, 1) / Range
