# -*- coding: utf-8 -*-
# jyoshida-sci 2015/08/31
import numpy as np
from scipy.optimize import fsolve

RANGEENERGY_MODEL = "Mishina"
# RANGEENERGY_MODEL = "ATIMA"
FIRST_IMPORT_ATIMA = True


def FirstRunForATIMA_RangeEnergy():
    global FIRST_IMPORT_ATIMA
    if FIRST_IMPORT_ATIMA:
        import sys

        sys.path.append("./ATIMA_RangeEnergy")
        import atima_rangeenergy

        atima_rangeenergy.ATIMA_splines = atima_rangeenergy.GetEmulsionSpline(
            "E07_standard"
        )
        FIRST_IMPORT_ATIMA = False


def RangeStragglingFromRange(Mass, Range, Z, density):
    """
    To get RangeStraggling from range

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
    if RANGEENERGY_MODEL == "Mishina":
        return RangeEnergyImpl().RangeStragglingFromRange(Mass, Range, Z, density)
    elif RANGEENERGY_MODEL == "ATIMA":
        FirstRunForATIMA_RangeEnergy()
        import atima_rangeenergy

        return atima_rangeenergy.RangeStragglingFromRange(Mass, Range, Z, density)
    else:
        raise


def RangeStragglingFromKE(Mass, KE, Z, density):
    """
    To RangeStraggling from KE

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
    if RANGEENERGY_MODEL == "Mishina":
        return RangeEnergyImpl().RangeStragglingFromKE(Mass, KE, Z, density)
    elif RANGEENERGY_MODEL == "ATIMA":
        FirstRunForATIMA_RangeEnergy()
        import atima_rangeenergy

        return atima_rangeenergy.RangeStragglingFromKE(Mass, KE, Z, density)
    else:
        raise


def KEfromRange(Mass, Range, Z, density):
    """
    To get KE from range

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
    if RANGEENERGY_MODEL == "Mishina":
        return RangeEnergyImpl().KEfromRange(Mass, Range, Z, density)
    elif RANGEENERGY_MODEL == "ATIMA":
        FirstRunForATIMA_RangeEnergy()
        import atima_rangeenergy

        return atima_rangeenergy.KEfromRange(Mass, Range, Z, density)
    else:
        raise


def RangeFromKE(Mass, KE, Z, density):
    """
    To range from KE

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
    if RANGEENERGY_MODEL == "Mishina":
        return RangeEnergyImpl().RangeFromKE(Mass, KE, Z, density)
    elif RANGEENERGY_MODEL == "ATIMA":
        FirstRunForATIMA_RangeEnergy()
        import atima_rangeenergy

        return atima_rangeenergy.RangeFromKE(Mass, KE, Z, density)
    else:
        raise


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
    if RANGEENERGY_MODEL == "Mishina":

        def Densityfunc(density):
            return RangeFromKE(Mass, KE, Z, density) - Range

        initial_guess = 3.0
        solution = fsolve(Densityfunc, initial_guess)
        return solution[0]
    elif RANGEENERGY_MODEL == "ATIMA":
        FirstRunForATIMA_RangeEnergy()
        import atima_rangeenergy

        return atima_rangeenergy.DensityFromKERange(Mass, KE, Range, Z)
    else:
        raise


Mp = 938.272  # proton mass


class RangeEnergyImpl:
    # constants
    Mp = 938.272  # proton mass
    LMp = np.log(Mp)  # log(proton mass)
    D0 = 3.815  # density of standard emulsion
    r = 0.884  # parameter for E07 emulsion, default_r=1.0

    # def __init__(self):

    # from Nkzw-san's Fortran-code
    # This function returns range in standard emulsion in micron units
    # @param input Mass[MeV] of a particle
    # @param input KE[MeV] of a particle
    def RangeInStandardEmulsionNk(self, Mass, KE):
        KEM = KE / Mass
        LKEM = np.log10(KEM)

        if KEM < 0.0001:
            return 479.210 * pow(KEM, 0.675899)
        else:
            Rs = 6.05595
            Rs += 1.38639 * LKEM
            Rs += -0.302838 * LKEM**2
            Rs += -0.0602134 * LKEM**3
            Rs += 0.0359347 * LKEM**4
            Rs += 0.0195023 * LKEM**5
            Rs += 0.00348314 * LKEM**6
            Rs += 0.000185264 * LKEM**7
            return 10.0**Rs

    # Mishina's fitting 2014
    # This function returns KE in standard emulsion in MeV units
    # @param input LnRange_mm
    # fitted by Mishina
    def ProtonKEfromRangeInStandardEmulsion_part1(self, LnRange_mm):
        LR = LnRange_mm
        LK = -2.288460778
        LK += +1.382747508 * LR
        LK += -0.439300692 * LR**2
        LK += +0.162697682 * LR**3
        LK += -0.037735480 * LR**4
        LK += +0.005152047 * LR**5
        LK += -0.000373872 * LR**6
        LK += +0.000010917 * LR**7
        LnKE_MeV = LK
        return LnKE_MeV

    # fitted by Mishina
    def ProtonKEfromRangeInStandardEmulsion_part2(self, LnRange_mm):
        LR = LnRange_mm
        LK = 12.499454326
        LK += -12.637449190 * LR
        LK += +5.296813187 * LR**2
        LK += -1.163641812 * LR**3
        LK += +0.151898030 * LR**4
        LK += -0.011803694 * LR**5
        LK += +0.000505820 * LR**6
        LK += -0.000009219 * LR**7
        LnKE_MeV = LK
        return LnKE_MeV

    # fitted by Mishina
    def ProtonKEfromRangeInStandardEmulsion_part3(self, LnRange_mm):
        LR = LnRange_mm
        LK = -0.52629642
        LK += +0.31555326 * LR
        LK += +0.021856192 * LR**2
        LK += +0.0012217823 * LR**3
        LK += -0.00026892371 * LR**4
        LK += +0.00001057489 * LR**5
        LnKE_MeV = LK
        return LnKE_MeV

    # scipy.optimize.fsolveの返り値がnp.ndarrayなので、solution[0]として実数にしている。
    def RangeInStandardEmulsion(self, Mass, KE):
        if KE <= 0.0:
            return 0.0
        KEM = KE / Mass
        MKEM = np.log(KE * self.Mp / Mass)
        Rs = 0
        if KEM < 0.0001:
            Rs = 479.210 * pow(KEM, 0.675899)
        elif MKEM < 1.930606146327:
            KEfunc = (
                lambda LnRange_mm: self.ProtonKEfromRangeInStandardEmulsion_part1(
                    LnRange_mm
                )
                - MKEM
            )
            initial_guess = 3.0
            solution = fsolve(KEfunc, initial_guess)
            Rs = np.exp(solution[0])
        elif MKEM < 4.405:
            KEfunc = (
                lambda LnRange_mm: self.ProtonKEfromRangeInStandardEmulsion_part2(
                    LnRange_mm
                )
                - MKEM
            )
            initial_guess = 6.0
            solution = fsolve(KEfunc, initial_guess)
            Rs = np.exp(solution[0])
        else:
            KEfunc = (
                lambda LnRange_mm: self.ProtonKEfromRangeInStandardEmulsion_part3(
                    LnRange_mm
                )
                - MKEM
            )
            initial_guess = 10.0
            solution = fsolve(KEfunc, initial_guess)
            Rs = np.exp(solution[0])
        return Rs

    # Mishina's original function
    def FunctionRs(self, KE, Mass):
        KEM = KE / Mass
        MKEM = np.log(KE * self.Mp / Mass)

        dd = 0.00001  # ;//step for italation

        if KEM < 0.0001:
            Rs = 479.210 * pow(KEM, 0.675899)

        elif MKEM < 1.930606146327:
            d0 = 3.0000
            y0 = self.Rs_function1(d0)
            while abs(MKEM - y0) > 0.00001:
                d0 = (d0 + dd) if (MKEM > y0) else (d0 - dd)
                y0 = self.Rs_function1(d0)
            Rs = np.exp(d0)

        elif MKEM < 4.405:
            d0 = 6.0000
            y0 = self.Rs_function2(d0)
            while abs(MKEM - y0) > 0.00001:
                d0 = (d0 + dd) if (MKEM > y0) else (d0 - dd)
                y0 = self.Rs_function2(d0)
            Rs = np.exp(d0)

        else:
            d0 = 10.0000
            y0 = self.Rs_function3(d0)
            while abs(MKEM - y0) > 0.00001:
                d0 = (d0 + dd) if (MKEM > y0) else (d0 - dd)
                y0 = self.Rs_function3(d0)
            Rs = np.exp(d0)

        return Rs

    # RsRwRatio fitted by Dr.Tovee and Dr.Gajewski
    # If the range is too short (< 1micron), this function returns constant.
    def FunctionRsRwRatio(self, Rs):
        rate = -0.107714711
        if Rs >= 1.0:
            LRs = np.log(Rs)
            rate += -0.332543998 * LRs
            rate += +0.141029694 * LRs**2
            rate += -0.044679440 * LRs**3
            rate += +0.008162611 * LRs**4
            rate += -0.000830409 * LRs**5
            rate += +0.000044038 * LRs**6
            rate += -0.000000951 * LRs**7
        return np.exp(rate)

    # Cz fitted by Mishina
    def FunctionCz(self, Z, beta):
        if abs(Z) == 1:
            return 0.0
        FX = 137.0 * beta / abs(Z)

        if FX <= 0.5:  # //regionI: a*FX^b
            return 0.168550736771407 * pow(FX, 1.90707106569386)
        elif FX <= 2.51:  # regionII: polinominal7
            val = 0.002624371
            val += -0.081622520 * FX
            val += +0.643381535 * FX**2
            val += -0.903648583 * FX**3
            val += +0.697505012 * FX**4
            val += -0.302935572 * FX**5
            val += +0.067662990 * FX**6
            val += -0.006004180 * FX**7
            return val
        else:  # regionIII: constant
            return 0.217598079611354

    # range in the standard emulsion -> our emulsion
    def DensityCorrectionFactor(self, emulsion_density, rs_rw_ratio):
        # wrong, modified formulae in 2014
        # factor = emulsion_density / self.D0 + ((self.r * (self.D0 - emulsion_density)) / (self.r * self.D0 - 1.0)) * rs_rw_ratio
        # formulae given by Heckman
        factor = (self.r * emulsion_density - 1) / (self.r * self.D0 - 1) + (
            (self.r * (self.D0 - emulsion_density)) / (self.r * self.D0 - 1.0)
        ) * rs_rw_ratio
        return factor

    # Energy->Range calculation
    def RangeFromKE(self, mass, KE, Z, emulsion_density):
        if KE <= 0.0:
            return 0.0
        # electron or positron
        if np.abs(mass - 0.511) < 0.001:
            # electron
            if Z < 0:
                # fitting 10-2400keV, Barkas p.444
                logKE = np.log10(KE)
                logR = 0.025388766 * logKE**5
                logR += 0.124110282 * logKE**4
                logR += 0.124652583 * logKE**3
                logR += -0.235688194 * logKE**2
                logR += 1.182432326 * logKE
                logR += 3.209584506
                range_standard = 10**logR
                # fitting NIST e-star
                # logR = -0.002353947 * logKE**5
                # logR = 0.012173926 * logKE**4
                # logR = 0.000226019 * logKE**3
                # logR = -0.241854689 * logKE**2
                # logR = 1.213878993 * logKE
                # logR = 3.213967752
            # positoron
            else:
                # fitting 10-2400keV, Barkas p.444
                logKE = np.log10(KE)
                logR = 0.012428679 * logKE**5
                logR += 0.059522101 * logKE**4
                logR += 0.020198153 * logKE**3
                logR += -0.296290819 * logKE**2
                logR += 1.211953980 * logKE
                logR += 3.215026031
                range_standard = 10**logR
            # correction for range
            rs_rw_ratio = self.FunctionRsRwRatio(range_standard)  # Rs/Rw ratio
            R = range_standard / self.DensityCorrectionFactor(
                emulsion_density, rs_rw_ratio
            )
        # other particles
        else:
            # range as proton in standard emulsion
            # range_standard = self.RangeInStandardEmulsionNk(mass, KE)#Nakazawa-san's fitting function
            range_standard = self.RangeInStandardEmulsion(
                mass, KE
            )  # Mishina's fitting function
            # correction for range
            rs_rw_ratio = self.FunctionRsRwRatio(range_standard)  # Rs/Rw ratio
            range_emulsion = range_standard / (
                self.DensityCorrectionFactor(emulsion_density, rs_rw_ratio)
            )  # range as proton in this emulsion
            # calculating Cz
            E = mass + KE  # total energy
            P = np.sqrt(E * E - mass * mass)  # momentum norm
            beta = P / E  # beta of particle
            Cz = self.FunctionCz(Z, beta)
            # correction factors
            CPS = 1
            CPM = 1
            CF = 1
            # Range
            R1 = CPS * (mass / self.Mp) / (Z * Z) * range_emulsion
            R2 = CPM * (mass / self.Mp) * pow(abs(Z), 2.0 / 3.0) * Cz  # R_ext
            R = (R1 + R2) / CF
        return R

    # This is the inverse-function of RangeFromKineticEnergy
    # scipy.optimize.fsolveの返り値がnp.ndarrayなので、solution[0]として実数にしている。
    def KEfromRange(self, Mass, Range, Z, densityEM):
        if Range <= 0.0:
            return 0.0

        Rfunc = lambda KE: self.RangeFromKE(Mass, KE, Z, densityEM) - Range
        initial_guess = 1.0  # any positive number
        solution = fsolve(Rfunc, initial_guess)
        if solution[0] < 0:
            raise Exception()
        return solution[0]

    def RangeStragglingFromKE(self, Mass, KE, Z, densityEM):
        KEM = KE / Mass
        M = Mass / self.Mp
        KEoverM = KE / M
        factorRangeStraggling = 0.0

        LogKEoverM = np.log10(KEoverM)
        if LogKEoverM < np.log10(2000 / M):
            factorRangeStraggling = (
                1.0164402329033
                * pow(10.0, 0.30754491034543 - 0.110592840462518 * LogKEoverM)
                / 100
            )
        else:
            factorRangeStraggling = (
                pow(10.0, -0.461423281494685 + 0.124489776442783 * LogKEoverM) / 100
            )

        dRp = 0
        if KEM < 0.0001:
            dRp = 479.210 * pow(KEoverM / self.Mp, 0.675899)
        else:
            # 多項式フィットしたものby Mishina
            val = 1.147272863
            val += 1.481654835 * LogKEoverM
            val += 0.156395018 * LogKEoverM**2
            val += -0.078039243 * LogKEoverM**3
            val += 0.065765281 * LogKEoverM**4
            val += -0.030414179 * LogKEoverM**5
            val += 0.005482754 * LogKEoverM**6
            val += -0.000336044 * LogKEoverM**7
            dRp = pow(10.0, val)

        return np.sqrt(M) / (Z * Z) * dRp * factorRangeStraggling

    def RangeStragglingFromRange(self, Mass, Range, Z, densityEM):
        if Range <= 0:
            return 0
        KE = self.KEfromRange(Mass, Range, Z, densityEM)
        return self.RangeStragglingFromKE(Mass, KE, Z, densityEM)
