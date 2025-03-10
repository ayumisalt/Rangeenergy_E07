# -*- coding: utf-8 -*-
"""
@author: ekawa, jyoshida
"""

import numpy as np
from uncertainties import ufloat
from uncertainties.umath import sin,cos,sqrt,fabs,acos,atan2
import re
import sys
import os
import itertools
import rangeenergy as reen

table = [
{'S':0,  'A':1 , 'Z':0 , 'M':939.565  , 'N':'n'},
{'S':0,  'A':1 , 'Z':1 , 'M':938.272  , 'N':'p'},
{'S':0,  'A':2 , 'Z':1 , 'M':1875.613 , 'N':'d'},
{'S':0,  'A':3 , 'Z':1 , 'M':2808.921 , 'N':'t'},
{'S':0,  'A':3 , 'Z':2 , 'M':2808.391 , 'N':'He3'},
{'S':0,  'A':4 , 'Z':1 , 'M':3750.086 , 'N':'H4'}, # decay
{'S':0,  'A':4 , 'Z':2 , 'M':3727.379 , 'N':'He4'},
{'S':0,  'A':5 , 'Z':2 , 'M':4667.680 , 'N':'He5'}, # decay
{'S':0,  'A':5 , 'Z':3 , 'M':4667.616 , 'N':'Li5'}, # decay
{'S':0,  'A':6 , 'Z':2 , 'M':5605.534 , 'N':'He6'},
{'S':0,  'A':6 , 'Z':3 , 'M':5601.518 , 'N':'Li6'},
{'S':0,  'A':6 , 'Z':4 , 'M':5605.296 , 'N':'Be6'}, # decay
{'S':0,  'A':7 , 'Z':2 , 'M':6545.510 , 'N':'He7'}, # decay
{'S':0,  'A':7 , 'Z':3 , 'M':6533.833 , 'N':'Li7'},
{'S':0,  'A':7 , 'Z':4 , 'M':6534.184 , 'N':'Be7'},
{'S':0,  'A':8 , 'Z':2 , 'M':7482.540 , 'N':'He8'},
{'S':0,  'A':8 , 'Z':3 , 'M':7471.365 , 'N':'Li8'},
{'S':0,  'A':8 , 'Z':4 , 'M':7454.851 , 'N':'Be8'}, # decay
{'S':0,  'A':8 , 'Z':5 , 'M':7472.320 , 'N':'B8'},
{'S':0,  'A':9 , 'Z':2 , 'M':8423.360 , 'N':'He9'}, # decay
{'S':0,  'A':9 , 'Z':3 , 'M':8406.868 , 'N':'Li9'},
{'S':0,  'A':9 , 'Z':4 , 'M':8392.751 , 'N':'Be9'},
{'S':0,  'A':9 , 'Z':5 , 'M':8393.309 , 'N':'B9'},
{'S':0,  'A':9 , 'Z':6 , 'M':8409.293 , 'N':'C9'},
{'S':0,  'A':10, 'Z':3 , 'M':9346.460 , 'N':'Li10'}, # decay
{'S':0,  'A':10, 'Z':4 , 'M':9325.504 , 'N':'Be10'},
{'S':0,  'A':10, 'Z':5 , 'M':9324.437 , 'N':'B10'},
{'S':0,  'A':10, 'Z':6 , 'M':9327.574 , 'N':'C10'},
{'S':0,  'A':11, 'Z':3 , 'M':10285.630, 'N':'Li11'},
{'S':0,  'A':11, 'Z':4 , 'M':10264.568, 'N':'Be11'},
{'S':0,  'A':11, 'Z':5 , 'M':10252.548, 'N':'B11'},
{'S':0,  'A':11, 'Z':6 , 'M':10254.019, 'N':'C11'},
{'S':0,  'A':12, 'Z':4 , 'M':11200.963, 'N':'Be12'},
{'S':0,  'A':12, 'Z':5 , 'M':11188.744, 'N':'B12'},
{'S':0,  'A':12, 'Z':6 , 'M':11174.864, 'N':'C12'},
{'S':0,  'A':12, 'Z':7 , 'M':11191.691, 'N':'N12'},
{'S':0,  'A':13, 'Z':4 , 'M':12141.038, 'N':'Be13'},
{'S':0,  'A':13, 'Z':5 , 'M':12123.430, 'N':'B13'},
{'S':0,  'A':13, 'Z':6 , 'M':12109.483, 'N':'C13'},
{'S':0,  'A':13, 'Z':7 , 'M':12111.193, 'N':'N13'},
{'S':0,  'A':13, 'Z':8 , 'M':12128.452, 'N':'O13'},
{'S':0,  'A':14, 'Z':4 , 'M':13078.827, 'N':'Be14'},
{'S':0,  'A':14, 'Z':5 , 'M':13062.026, 'N':'B14'},
{'S':0,  'A':14, 'Z':6 , 'M':13040.872, 'N':'C14'},
{'S':0,  'A':14, 'Z':7 , 'M':13040.204, 'N':'N14'},
{'S':0,  'A':14, 'Z':8 , 'M':13044.839, 'N':'O14'},
{'S':0,  'A':15, 'Z':5 , 'M':13998.815, 'N':'B15'},
{'S':0,  'A':15, 'Z':6 , 'M':13979.219, 'N':'C15'},
{'S':0,  'A':15, 'Z':7 , 'M':13968.936, 'N':'N15'},
{'S':0,  'A':15, 'Z':8 , 'M':13971.181, 'N':'O15'},
{'S':0,  'A':16, 'Z':5 , 'M':14938.463, 'N':'B16'},
{'S':0,  'A':16, 'Z':6 , 'M':14914.534, 'N':'C16'},
{'S':0,  'A':16, 'Z':7 , 'M':14906.013, 'N':'N16'},
{'S':0,  'A':16, 'Z':8 , 'M':14895.082, 'N':'O16'},
{'S':0,  'A':17, 'Z':5 , 'M':15876.561, 'N':'B17'},
{'S':0,  'A':17, 'Z':6 , 'M':15853.366, 'N':'C17'},
{'S':0,  'A':17, 'Z':7 , 'M':15839.694, 'N':'N17'},
{'S':0,  'A':17, 'Z':8 , 'M':15830.504, 'N':'O17'},

{'S':0,  'A':18, 'Z':5 , 'M':16816.132, 'N':'B18'},
{'S':0,  'A':18, 'Z':6 , 'M':16788.747, 'N':'C18'},
{'S':0,  'A':18, 'Z':7 , 'M':16776.431, 'N':'N18'},
{'S':0,  'A':18, 'Z':8 , 'M':16762.024, 'N':'O18'},
{'S':0,  'A':19, 'Z':5 , 'M':17755.603, 'N':'B19'},
{'S':0,  'A':19, 'Z':6 , 'M':17727.736, 'N':'C19'},
{'S':0,  'A':19, 'Z':7 , 'M':17710.667, 'N':'N19'},
{'S':0,  'A':19, 'Z':8 , 'M':17697.634, 'N':'O19'},

{'S':0,  'A':2 , 'Z':0 , 'M':939.565*2, 'N':'2n'},
{'S':0,  'A':3 , 'Z':0 , 'M':939.565*3, 'N':'3n'},
{'S':0,  'A':0 , 'Z':-1, 'M':139.570  , 'N':'pi-'},
{'S':0,  'A':0 , 'Z':0 , 'M':134.977  , 'N':'pi0'},
{'S':0,  'A':1 , 'Z':0 , 'M':134.977 + 939.565   ,'N':'pi0 + n'},
{'S':0,  'A':2 , 'Z':0 , 'M':134.977 + 939.565*2 ,'N':'pi0 + 2n'},

{'S':-1, 'A':1 , 'Z':0 , 'M':1115.683 , 'N':'L'},
{'S':-1, 'A':2 , 'Z':0 , 'M':1115.683 + 939.565,  'N':'L + n'},
{'S':-1, 'A':3 , 'Z':0 , 'M':1115.683 + 939.565*2,'N':'L + 2n'},

{'S':-1, 'A':3 , 'Z':1, 'M':2991.166 , 'N':'H3L'},
{'S':-1, 'A':4 , 'Z':1, 'M':3922.564 , 'N':'H4L'},
{'S':-1, 'A':4 , 'Z':2, 'M':3921.684 , 'N':'He4L'},
{'S':-1, 'A':5 , 'Z':2, 'M':4839.942 , 'N':'He5L'},
{'S':-1, 'A':6 , 'Z':2, 'M':5779.183 , 'N':'He6L'},
{'S':-1, 'A':6 , 'Z':3, 'M':5778.799 , 'N':'Li6L'},
{'S':-1, 'A':7 , 'Z':2, 'M':6715.667 , 'N':'He7L'},
{'S':-1, 'A':7 , 'Z':3, 'M':6711.621 , 'N':'Li7L'},
{'S':-1, 'A':7 , 'Z':4, 'M':6715.819 , 'N':'Be7L'},
{'S':-1, 'A':8 , 'Z':2, 'M':7654.033 , 'N':'He8L'},
{'S':-1, 'A':8 , 'Z':3, 'M':7642.716 , 'N':'Li8L'},
{'S':-1, 'A':8 , 'Z':4, 'M':7643.027 , 'N':'Be8L'},
{'S':-1, 'A':9 , 'Z':3, 'M':8578.548 , 'N':'Li9L'},
{'S':-1, 'A':9 , 'Z':4, 'M':8563.824 , 'N':'Be9L'},
{'S':-1, 'A':9 , 'Z':5, 'M':8579.713 , 'N':'B9L'},
{'S':-1, 'A':10, 'Z':4, 'M':9499.324 , 'N':'Be10L'},
{'S':-1, 'A':10, 'Z':5, 'M':9500.102 , 'N':'B10L'},
{'S':-1, 'A':11, 'Z':5, 'M':10429.880, 'N':'B11L'},
{'S':-1, 'A':12, 'Z':5, 'M':11356.861, 'N':'B12L'},
{'S':-1, 'A':12, 'Z':6, 'M':11358.902, 'N':'C12L'},
{'S':-1, 'A':13, 'Z':6, 'M':12278.857, 'N':'C13L'},
{'S':-1, 'A':14, 'Z':6, 'M':13212.996, 'N':'C14L'},
{'S':-1, 'A':14, 'Z':7, 'M':13214.706, 'N':'N14L'},
{'S':-1, 'A':15, 'Z':7, 'M':14142.297, 'N':'N15L'},
{'S':-1, 'A':16, 'Z':7, 'M':15070.859, 'N':'N16L'},
{'S':-1, 'A':16, 'Z':8, 'M':15074.444, 'N':'O16L'},

{'S':-1, 'A':5 , 'Z':1, 'M':4861.819 , 'N':'H5L'},
{'S':-1, 'A':9 , 'Z':2, 'M':8591.624 , 'N':'He9L'},
{'S':-1, 'A':10, 'Z':2, 'M':9531.587 , 'N':'He10L'},
{'S':-1, 'A':10, 'Z':3, 'M':9513.133 , 'N':'Li10L'},
{'S':-1, 'A':11, 'Z':3, 'M':10451.438, 'N':'Li11L'},
{'S':-1, 'A':11, 'Z':4, 'M':10432.999, 'N':'Be11L'},
{'S':-1, 'A':12, 'Z':4, 'M':11371.443, 'N':'Be12L'},
{'S':-1, 'A':13, 'Z':5, 'M':12291.946, 'N':'B13L'},
{'S':-1, 'A':14, 'Z':5, 'M':13225.508, 'N':'B14L'},
{'S':-1, 'A':11, 'Z':6, 'M':10433.154, 'N':'C11L'},
{'S':-1, 'A':15, 'Z':6, 'M':14143.397, 'N':'C15L'},
{'S':-1, 'A':16, 'Z':6, 'M':15080.980, 'N':'C16L'},
{'S':-1, 'A':13, 'Z':7, 'M':12295.161, 'N':'N13L'},
{'S':-1, 'A':17, 'Z':7, 'M':16007.230, 'N':'N17L'},
{'S':-1, 'A':18, 'Z':7, 'M':16940.348, 'N':'N18L'},

{'S':-2, 'A':4 , 'Z':1, 'M':4106.719 , 'N':'H4LL'},
{'S':-2, 'A':5 , 'Z':1, 'M':5036.207 , 'N':'H5LL'},
{'S':-2, 'A':6 , 'Z':1, 'M':5973.552 , 'N':'H6LL'},
{'S':-2, 'A':5 , 'Z':2, 'M':5034.977 , 'N':'He5LL'},
{'S':-2, 'A':6 , 'Z':2, 'M':5952.505 , 'N':'He6LL'},
{'S':-2, 'A':7 , 'Z':2, 'M':6890.686 , 'N':'He7LL'},
{'S':-2, 'A':8 , 'Z':2, 'M':7825.800 , 'N':'He8LL'},
{'S':-2, 'A':9 , 'Z':2, 'M':8762.556 , 'N':'He9LL'},
{'S':-2, 'A':10, 'Z':2, 'M':9700.708 , 'N':'He10LL'},
{'S':-2, 'A':11, 'Z':2, 'M':10639.814, 'N':'He11LL'},
{'S':-2, 'A':7,  'Z':3, 'M':6889.982 , 'N':'Li7LL'},
{'S':-2, 'A':8,  'Z':3, 'M':7821.724 , 'N':'Li8LL'},
{'S':-2, 'A':9,  'Z':3, 'M':8751.599 , 'N':'Li9LL'},
{'S':-2, 'A':10, 'Z':3, 'M':9685.731 , 'N':'Li10LL'},
{'S':-2, 'A':11, 'Z':3, 'M':10619.398, 'N':'Li11LL'},
{'S':-2, 'A':12, 'Z':3, 'M':11556.416, 'N':'Li12LL'},
{'S':-2, 'A':8,  'Z':4, 'M':7826.342 , 'N':'Be8LL'},
{'S':-2, 'A':9,  'Z':4, 'M':8751.870 , 'N':'Be9LL'},
{'S':-2, 'A':10, 'Z':4, 'M':9672.797 , 'N':'Be10LL'},
{'S':-2, 'A':11, 'Z':4, 'M':10605.897, 'N':'Be11LL'},
{'S':-2, 'A':12, 'Z':4, 'M':11540.494, 'N':'Be12LL'},
{'S':-2, 'A':13, 'Z':4, 'M':12478.318, 'N':'Be13LL'},
{'S':-2, 'A':10, 'Z':5, 'M':9687.106 , 'N':'B10LL'},
{'S':-2, 'A':11, 'Z':5, 'M':10606.895, 'N':'B11LL'},
{'S':-2, 'A':12, 'Z':5, 'M':11535.323, 'N':'B12LL'},
{'S':-2, 'A':13, 'Z':5, 'M':12461.174, 'N':'B13LL'},
{'S':-2, 'A':14, 'Z':5, 'M':13395.148, 'N':'B14LL'},
{'S':-2, 'A':15, 'Z':5, 'M':14327.586, 'N':'B15LL'},
{'S':-2, 'A':12, 'Z':6, 'M':11538.734, 'N':'C12LL'},
{'S':-2, 'A':13, 'Z':6, 'M':12463.785, 'N':'C13LL'},
{'S':-2, 'A':14, 'Z':6, 'M':13382.850, 'N':'C14LL'},
{'S':-2, 'A':15, 'Z':6, 'M':14316.509, 'N':'C15LL'},
{'S':-2, 'A':16, 'Z':6, 'M':15245.922, 'N':'C16LL'},
{'S':-2, 'A':17, 'Z':6, 'M':16182.741, 'N':'C17LL'},
{'S':-2, 'A':14, 'Z':7, 'M':13398.631, 'N':'N14LL'},
{'S':-2, 'A':15, 'Z':7, 'M':14318.219, 'N':'N15LL'},
{'S':-2, 'A':16, 'Z':7, 'M':15244.390, 'N':'N16LL'},
{'S':-2, 'A':17, 'Z':7, 'M':16172.782, 'N':'N17LL'},
{'S':-2, 'A':18, 'Z':7, 'M':17108.447, 'N':'N18LL'},
{'S':-2, 'A':19, 'Z':7, 'M':18041.002, 'N':'N19LL'},

{'S':-2, 'A':13, 'Z':5, 'M':11174.864 + 1321.71, 'N':'Xi- & C12'},
{'S':-2, 'A':14, 'Z':5, 'M':12109.483 + 1321.71, 'N':'Xi- & C13'},
{'S':-2, 'A':15, 'Z':6, 'M':13040.204 + 1321.71, 'N':'Xi- & N14'},
{'S':-2, 'A':16, 'Z':6, 'M':13968.936 + 1321.71, 'N':'Xi- & N15'},
{'S':-2, 'A':17, 'Z':7, 'M':14895.082 + 1321.71, 'N':'Xi- & O16'},
{'S':-2, 'A':18, 'Z':7, 'M':15830.504 + 1321.71, 'N':'Xi- & O17'},
{'S':-2, 'A':19, 'Z':7, 'M':16762.024 + 1321.71, 'N':'Xi- & O18'},

{'S':-1, 'A':13, 'Z':5, 'M':11174.864 + 1197.449, 'N':'Sigma- & C12'},
{'S':-1, 'A':14, 'Z':5, 'M':12109.483 + 1197.449, 'N':'Sigma- & C13'},
{'S':-1, 'A':15, 'Z':6, 'M':13040.204 + 1197.449, 'N':'Sigma- & N14'},
{'S':-1, 'A':16, 'Z':6, 'M':13968.936 + 1197.449, 'N':'Sigma- & N15'},
{'S':-1, 'A':17, 'Z':7, 'M':14895.082 + 1197.449, 'N':'Sigma- & O16'},
{'S':-1, 'A':18, 'Z':7, 'M':15830.504 + 1197.449, 'N':'Sigma- & O17'},
{'S':-1, 'A':19, 'Z':7, 'M':16762.024 + 1197.449, 'N':'Sigma- & O18'},
]

def ke2mom(mass, KE):
    return np.sqrt((mass+KE)**2 - mass**2)

class Particle():
    def __init__(self, range_micron, range_err, theta_deg, theta_err, phi_deg, phi_err, Sflag):
        # geometrical values, type: ufloat
        self.range = ufloat(range_micron, range_err)
        self.theta_deg = ufloat(theta_deg, theta_err)
        self.phi_deg = ufloat(phi_deg, phi_err)
        self.theta = self.theta_deg * np.pi/180
        self.phi = self.phi_deg * np.pi/180
        self.Sflag = Sflag
    def calc_kinematics(self, N, density):
        rs = reen.RangeStragglingFromRange(N['M'], self.range.n, N['Z'], density)
        range_c = ufloat(self.range.n, sqrt(self.range.s**2 + rs**2))
        ke_c    = reen.KEfromRange(N['M'], self.range.n, N['Z'], density)
        mom_c   = ke2mom(N['M'], ke_c)
        range_l = range_c.n - range_c.s
        ke_l    = reen.KEfromRange(N['M'], range_l, N['Z'], density)
        mom_l   = ke2mom(N['M'], ke_l)
        range_r = range_c.n + range_c.s
        ke_r    = reen.KEfromRange(N['M'], range_r, N['Z'], density)
        mom_r   = ke2mom(N['M'], ke_r)
        self.range_c = range_c
        self.KE  = ufloat(ke_c, np.fabs(ke_l - ke_r)/2.0)
        self.p = ufloat(mom_c, np.fabs(mom_l - mom_r)/2.0)
        self.mass = ufloat(N['M'], 0.0)
        self.energy = self.mass + self.KE
    def momentum(self, axis):
        if axis=="x" or axis==0:
            return self.p * sin(self.theta) * cos(self.phi)
        elif axis=="y" or axis==1:
            return self.p * sin(self.theta) * sin(self.phi)
        elif axis=="z" or axis==2:
            return self.p * cos(self.theta)
        else:
            raise Exception("invalid axis index")
    def roundH_rounda(self, i_constraint, val):
        if i_constraint==0 and val==0:
            return self.p.n / np.sqrt(self.mass.n**2 + self.p.n**2)
        elif i_constraint==1 and val==0:
            return np.sin(self.theta.n) * np.cos(self.phi.n)
        elif i_constraint==2 and val==0:
            return np.sin(self.theta.n) * np.sin(self.phi.n)
        elif i_constraint==3 and val==0:
            return np.cos(self.theta.n)
        elif i_constraint==0 and val==1:
            return 0.0
        elif i_constraint==1 and val==1:
            return self.p.n * np.cos(self.theta.n) * np.cos(self.phi.n)
        elif i_constraint==2 and val==1:
            return self.p.n * np.cos(self.theta.n) * np.sin(self.phi.n)
        elif i_constraint==3 and val==1:
            return self.p.n * -np.sin(self.theta.n)
        elif i_constraint==0 and val==2:
            return 0.0
        elif i_constraint==1 and val==2:
            return self.p.n * np.sin(self.theta.n) * -np.sin(self.phi.n)
        elif i_constraint==2 and val==2:
            return self.p.n * np.sin(self.theta.n) * np.cos(self.phi.n)
        elif i_constraint==3 and val==2:
            return 0.0
        else:
            raise Exception("invalid constraint index")
    def print_params(self):
        print("|mom|: {0:.2f} [MeV/c]".format(self.p))
        print("theta: {0:.4f} [rad]".format(self.theta))
        print("phi: {0:.4f} [rad]".format(self.phi))
        print("mass: {0:.2f} [MeV/c]".format(self.mass))
        print("energy: {0:.2f} [MeV/c]".format(self.energy))


class Particle2(Particle):
    def __init__(self, mom_MeV, mom_err, theta_rad, theta_err, phi_rad, phi_err, mass):
        # geometrical values, type: ufloat
        self.p = ufloat(mom_MeV, mom_err)
        self.theta = ufloat(theta_rad, theta_err)
        self.phi = ufloat(phi_rad, phi_err)
        self.theta_deg = self.theta * 180/np.pi
        self.phi_deg = self.phi * 180/np.pi
        self.mass = ufloat(mass, 0.0)
        self.KE = sqrt(self.mass**2 + self.p**2) - self.mass
        self.energy = self.mass + self.KE


class ParticleN(Particle):
    def __init__(self, px, py, pz, mass):
        # geometrical values, type: ufloat
        self.px = ufloat(px.n, px.s)
        self.py = ufloat(py.n, py.s)
        self.pz = ufloat(pz.n, pz.s)
        self.mass = ufloat(mass, 0.0)
        self.p = sqrt(self.px**2 + self.py**2 + self.pz**2)
        self.KE = sqrt(self.mass**2 + self.p**2) - self.mass
        self.theta = acos( self.pz/self.p)
        self.phi = atan2( self.py,  self.px)
        self.theta_deg = self.theta * 180/np.pi
        self.phi_deg = self.phi * 180/np.pi
        self.energy = self.mass + self.KE


# to search for nuclei
def Xi_bound_systems():
    return [x for x in table if re.search("^Xi- &", x["N"])]

def Sigma_bound_systems():
    return [x for x in table if re.search("^Sigma- &", x["N"])]

def double_Lambdas():
    return [x for x in table if re.search("LL$", x["N"])]

def single_Lambdas():
    return [x for x in table if re.search("L$", x["N"]) and x["Z"] > 0 and x["S"] == -1]

def normal_nuclei():
    return [x for x in table if x["S"] == 0 and x["A"] > 0 and not re.search("n$", x["N"])]

def neutrals():
    return [x for x in table if x["Z"] == 0]

def chargeds():
    return [x for x in table if x["Z"] != 0]

def non_strangeness_chargeds():
    return [x for x in table if x["S"] == 0 and x["Z"] != 0]

def get_particle_from_table(name):
    particles = [x for x in table if x["N"] == name]
    if len(particles) == 1:
        return particles[0]
    else:
        raise Exception("invalid name")

def get_particle_from_SAZ(s, a, z):
    particles = [x for x in table if x["S"] == s and x["A"] == a and x["Z"] == z]
    if len(particles) == 1:
        return particles[0]
    else:
        raise Exception("invalid particle")

def get_BLL(N):
    if N['S'] == -2:
        parent_mass = ufloat(get_particle_from_SAZ(0, N['A']-2, N['Z'])['M'], 0.0)
        lambda_mass = ufloat(get_particle_from_table('L')['M'], 0.0)
        double_mass = ufloat(N['M'], 0.0)
        return parent_mass + lambda_mass*2 - double_mass
    else:
        raise Exception("invalid particle")


# to select valid combination
def check_conservation(combi, key):
    buf = 0
    for i in range(1,len(combi)):
        buf += combi[i][key]
    return buf == combi[0][key]

def mass_number_conservation(combi):
    for i in range(1,len(combi)):
        if combi[0]["A"] < combi[i]["A"]:
            return False
    buf = 0
    for i in range(1,len(combi)):
        buf += combi[i]["A"]
    return buf == combi[0]["A"]

def charge_conservation(combi):
    return check_conservation(combi, "Z")

def strangeness_conservation(combi):
    return check_conservation(combi, "S")

def one_strangeness_decay(combi):
    buf = 0
    for i in range(1,len(combi)):
        buf += combi[i]["S"]
    return buf - combi[0]["S"] == 1

def Qvalue(combi):
    val = combi[0]["M"]
    for i in range(1,len(combi)):
        val -= combi[i]["M"]
    return val

def print_combination(combi):
    tmp = "{0} -> {1}".format(combi[0]["N"], combi[1]["N"])
    for i in range(2,len(combi)):
        tmp += " + {0}".format(combi[i]["N"])
    print(tmp)

def solve_kinematic_fit(eta0, V_eta0, D, d, E):
    # Applied Fitting Theory I General Least Squares Theory
    # 6. Solving for Unknown Parameters in the Constraint Equations
    # 1. Direct solution
    V_D = np.linalg.inv( D.dot(V_eta0.dot(D.transpose())) )
    V_E = np.linalg.inv( E.transpose().dot(V_D.dot(E)) )
    V_z = V_E
    V_lambda = V_D - V_D.dot(E.dot( V_E.dot( E.transpose().dot(V_D))))
    lambda0 = V_D.dot(d)
    z = -V_E.dot( E.transpose().dot(lambda0))
    lmbda = lambda0 + V_D.dot(E.dot(z))
    chisquare = lmbda.transpose().dot(np.linalg.inv(V_D).dot(lmbda))[0,0]
    V_eta =  V_eta0 - V_eta0.dot(D.transpose().dot(V_lambda.dot(D.dot(V_eta0))))
    eta = eta0 - V_eta0.dot(D.transpose().dot(lmbda))
    return V_D, V_z, V_lambda, V_eta, z, lmbda, eta, chisquare,

def total_momentum(particles, axis):
    mom = ufloat(0,0)
    for p in particles:
        tmp = p.momentum(axis)
        #これをかませないと誤差計算がおかしくなるというuncertaintiesのバグ(?)
        mom += ufloat(tmp.n, tmp.s)
    return mom

def total_energy(particles):
    ene = ufloat(0,0)
    for p in particles:
        tmp = p.energy
        ene += ufloat(tmp.n, tmp.s)
    return ene

def total_kinetic_energy(particles):
    KE = ufloat(0,0)
    for p in particles:
        tmp = p.KE
        KE += ufloat(tmp.n, tmp.s)
    return KE


def get_str_daughter_vals(daughters):
    mystr = ""
    for i,d in enumerate(daughters):
        i += 1#テキスト出力では番号1から
        mystr += "\n"
        mystr += "KE{0} = {1:9.3f} +/- {2:9.3f}\n".format(i, d.KE.n, d.KE.s)
        mystr += "p{0} = {1:9.3f} +/- {2:9.3f}\n".format(i, d.p.n, d.p.s)
        mystr += "  p{0}x = {1:9.3f} +/- {2:9.3f}\n".format(i, d.momentum(0).n, d.momentum(0).s)
        mystr += "  p{0}y = {1:9.3f} +/- {2:9.3f}\n".format(i, d.momentum(1).n, d.momentum(1).s)
        mystr += "  p{0}z = {1:9.3f} +/- {2:9.3f}\n".format(i, d.momentum(2).n, d.momentum(2).s)
        mystr += "R{0} = {1:9.3f} +/- {2:9.3f}\n".format(i, d.range_c.n, d.range_c.s)
    return mystr


def get_str_daughter_before_after(daughters, new_daughters):
    mystr = ""
    for i in range(len(daughters)):
        d0 = daughters[i]
        d1 = new_daughters[i]
        mystr += "\n"
        mystr += "p{0}     : {1:9.3f} +/- {2:6.3f}  -> {3:9.3f} +/- {4:6.3f}\n" \
                .format(i+1, d0.p.n, d0.p.s, d1.p.n, d1.p.s)
        mystr += "theta{0} : {1:9.3f} +/- {2:6.3f}  -> {3:9.3f} +/- {4:6.3f}\n" \
                .format(i+1, d0.theta_deg.n, d0.theta_deg.s, d1.theta_deg.n, d1.theta_deg.s)
        mystr += "phi{0}   : {1:9.3f} +/- {2:6.3f}  -> {3:9.3f} +/- {4:6.3f}\n" \
                .format(i+1, d0.phi_deg.n, d0.phi_deg.s, d1.phi_deg.n, d1.phi_deg.s)
    return mystr


def evaluate_kinematics(event_type, track, N, params, fall, fpos):

    cut_sig  = params["cut_sig"]
    max_dBLL = params["max_dBLL"]
    density =  params["density"]
    Q = Qvalue(N)

    for i in range(len(track)):
        track[i].calc_kinematics(N[i], density)

    parent = track[0]
    daughters = track[1:]
    n_track = len(daughters)

    # vals of system in ufloats
    total_final_energy = total_energy(daughters)
    total_px = total_momentum(daughters, "x")
    total_py = total_momentum(daughters, "y")
    total_pz = total_momentum(daughters, "z")
    total_p = sqrt(total_px**2 + total_py**2 + total_pz**2)
    if total_p.n == 0:
        total_p = ufloat( 0., np.sqrt(total_px.s**2 + total_py.s**2 + total_pz.s**2) )
    sigma_mom = sqrt( (total_px.n/total_px.s)**2 + (total_py.n/total_py.s)**2 + (total_pz.n/total_pz.s)**2 )
    invariant_mass_sq = total_final_energy**2  - total_p**2
    invariant_mass = sqrt(invariant_mass_sq)
    mass_gap = parent.mass - invariant_mass
    total_ke = total_kinetic_energy(daughters)
    QQ = Q - total_ke
    dBLL = ufloat(0, 0)
    if event_type == 1 or event_type == 3:
        dBLL = parent.mass - invariant_mass
    if event_type == 2:
        dBLL = invariant_mass - parent.mass
    BLL  = ufloat(0, 0)
    if event_type == 1:
        BLL = get_BLL(N[0]) + dBLL
    if event_type == 2:
        BLL = get_BLL(N[1]) + dBLL

    ##  kinematic fitting  #############################################
    # eta0-matrix: input value
    eta0 = np.zeros((n_track*3, 1))
    for i,d in enumerate(daughters):
        eta0[i*3+0,0] = d.p.n
        eta0[i*3+1,0] = d.theta.n
        eta0[i*3+2,0] = d.phi.n

    # constraints derivation matrix
    D = np.zeros((4, n_track*3))
    for i,d in enumerate(daughters):
        for c in range(4):# constraint
            for val in range(3):# val; 0:mom, 1:theta, 2:phi
                D[c][i*3+val] = d.roundH_rounda(c, val)

    # variance-covariance matrix
    V_eta0 = np.zeros((n_track*3, n_track*3))

    for i,d in enumerate(daughters):
        V_eta0[i*3+0,i*3+0] = d.p.s**2
        V_eta0[i*3+1,i*3+1] = d.theta.s**2
        V_eta0[i*3+2,i*3+2] = d.phi.s**2

    # constraints
    d = np.array([
        [total_final_energy.n - parent.mass.n],
        [total_px.n],
        [total_py.n],
        [total_pz.n]
    ])

    # unknown parameter
    E = np.zeros(( 4, 1 ))
    if event_type == 1 or event_type == 3 or event_type == 0:
        E[0,0] = -1
    if event_type == 2:
        #daughters[0] is double candidate
        E[0,0] = daughters[0].mass.n / np.hypot( daughters[0].mass.n, daughters[0].p.n)

    # solve kinematic fit
    V_D, V_z, V_lambda, V_eta, z, lmbda, eta, chisquare =  solve_kinematic_fit(eta0, V_eta0, D, d, E)

    # new vals
    new_daughters = []
    for i in range(n_track):
        new_daughters.append(
                Particle2(eta[i*3+0,0], np.sqrt(V_eta[i*3+0,i*3+0]),
                          eta[i*3+1,0], np.sqrt(V_eta[i*3+1,i*3+1]),
                          eta[i*3+2,0], np.sqrt(V_eta[i*3+2,i*3+2]),
                          daughters[i].mass.n))

    new_total_final_energy = total_energy(new_daughters)
    new_total_px = total_momentum(new_daughters, "x")
    new_total_py = total_momentum(new_daughters, "y")
    new_total_pz = total_momentum(new_daughters, "z")
    new_total_p = sqrt(new_total_px**2 + new_total_py**2 + new_total_pz**2)
    if new_total_p.n == 0:
        new_total_p = ufloat( 0., np.sqrt(new_total_px.s**2 + new_total_py.s**2 + new_total_pz.s**2) * 1.05 )
    new_invariant_mass_sq = new_total_final_energy**2  - new_total_p**2
    new_invariant_mass = sqrt(new_invariant_mass_sq)

    new_err_m = np.sqrt(V_z[0,0])
    if 'Xi-'  in N[0]['N']:
        new_err_m = sqrt(new_err_m**2 + 0.07**2)# Xi mass error = 0.07MeV/v^2 (PDG)
    new_dBLL  = ufloat(-1 * z[0,0], new_err_m)
    new_BLL   = ufloat(0, 0)
    if event_type == 1:
        new_BLL = get_BLL(N[0]) + new_dBLL
    if event_type == 2:
        new_BLL = get_BLL(N[1]) + new_dBLL

    ####################################################################
    # output
    mystr  = "{0} -> {1}".format(N[0]['N'], N[1]['N'])
    for i in range(2,n_track+1):
        mystr  += " + {0}".format(N[i]['N'])
    if event_type == 1:
        mystr  += "  : dBLL = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    if event_type == 2:
        mystr  += "  : dBLL - BXi = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    if event_type == 3:
        mystr  += "  : BXi = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    mystr  += "  ************************\n"

    mystr += get_str_daughter_vals(daughters)

    mystr += "\n"
    mystr += "     Calculated_Mass = {0:9.3f} +/- {1:9.3f}\n".format(invariant_mass.n, invariant_mass.s)
    mystr += "      Estimated_Mass = {0:9.3f}\n".format(N[0]['M'])
    mystr += "            Mass_gap = {0:9.3f}\n".format(mass_gap.n)
    #mystr += "         Signigfance = {0:9.3f} sigmas\n".format(Mass_gap.n/Calculated_Mass.s)
    mystr += "           Total_ene = {0:9.3f} +/- {1:9.3f}\n".format(total_final_energy.n, total_final_energy.s)
    mystr += "           Total_mom = {0:9.3f} +/- {1:9.3f}\n".format(total_p.n, total_p.s)
    mystr += "           Sigma_mom = {0:9.3f}\n".format(sigma_mom)
    mystr += "             Total_momx = {0:9.3f} +/- {1:9.3f}\n".format(total_px.n, total_px.s)
    mystr += "             Total_momy = {0:9.3f} +/- {1:9.3f}\n".format(total_py.n, total_py.s)
    mystr += "             Total_momz = {0:9.3f} +/- {1:9.3f}\n".format(total_pz.n, total_pz.s)
    mystr += "                   Q = {0:9.3f}\n".format(Q)
    mystr += "Total_Kinetic_Energy = {0:9.3f} +/- {1:9.3f}\n".format(total_ke.n, total_ke.s)
    mystr += "                  QQ = {0:9.3f} +/- {1:9.3f}\n".format(fabs(total_ke.n-Q), total_ke.s)
    mystr += "                dBLL = {0:9.3f} +/- {1:9.3f}\n".format(dBLL.n, dBLL.s)
    mystr += "                BLL  = {0:9.3f} +/- {1:9.3f}\n\n".format(BLL.n, BLL.s)
    mystr += "==== Kinematic Fitting ====\n"

    mystr += get_str_daughter_before_after(daughters, new_daughters)

    mystr += "\n"
    mystr += "Total_mom  : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_p.n, total_p.s, new_total_p.n, new_total_p.s)
    mystr += "Total_momx : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_px.n, total_px.s, new_total_px.n, new_total_px.s)
    mystr += "Total_momy : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_py.n, total_py.s, new_total_py.n, new_total_py.s)
    mystr += "Total_momz : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n\n" \
            .format(total_pz.n, total_pz.s, new_total_pz.n, new_total_pz.s)
    mystr += "dBLL       : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(dBLL.n, dBLL.s, new_dBLL.n, new_dBLL.s)
    mystr += "BLL        : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n\n\n" \
            .format(BLL.n, BLL.s, new_BLL.n, new_BLL.s)

    fall.write(mystr)

    if event_type == 0:
        if fabs(mass_gap) <= invariant_mass.s * cut_sig and \
        sigma_mom <= 3.78 and \
        fabs(QQ) <= total_ke.s * cut_sig:
            fpos.write(mystr)
    if event_type == 1 or event_type == 3:
        if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
        mass_gap - max_dBLL <= invariant_mass.s * cut_sig and \
        sigma_mom <= 3.78 and \
        QQ + max_dBLL >= total_ke.s * -cut_sig and \
        QQ - max_dBLL <= total_ke.s * cut_sig and \
        fabs(dBLL) < max_dBLL:
            fpos.write(mystr)
    if event_type == 2:
        if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
        mass_gap - max_dBLL <= invariant_mass.s * cut_sig and \
        sigma_mom <= 3.78 and \
        QQ + max_dBLL >= total_ke.s*-cut_sig and \
        QQ - max_dBLL <= total_ke.s*cut_sig and \
        fabs(dBLL) < max_dBLL:
            fpos.write(mystr)


def evaluate_kinematics_with_neutral(event_type, track, N, params, fallN, fposN):
    # 配列Nは配列trackの要素数プラス1．neutral候補が末尾に入っている。
    cut_sig  = params["cut_sig"]
    max_dBLL = params["max_dBLL"]
    density =  params["density"]

    Q = Qvalue(N)

    for i in range(len(track)):
        track[i].calc_kinematics(N[i], density)

    neutral = ParticleN(-total_momentum(track[1:], "x"), -total_momentum(track[1:], "y"), -total_momentum(track[1:], "z"), N[-1]['M'])

    parent = track[0]
    daughters = track[1:]
    n_track = len(daughters)

    # vals of system in ufloats
    total_final_energy = total_energy(daughters) + neutral.energy
    total_px = total_momentum(daughters, "x") + neutral.px
    total_py = total_momentum(daughters, "y") + neutral.py
    total_pz = total_momentum(daughters, "z") + neutral.pz
    total_p = sqrt(total_px**2 + total_py**2 + total_pz**2)
    if np.fabs(total_p.n) < 0.001:
        total_p = ufloat( 0., np.sqrt(total_px.s**2 + total_py.s**2 + total_pz.s**2) )
    invariant_mass_sq = total_final_energy**2  - total_p**2
    invariant_mass = sqrt(invariant_mass_sq)
    mass_gap = parent.mass - invariant_mass
    total_ke = total_kinetic_energy(daughters) + neutral.KE
    QQ = Q - total_ke
    dBLL = ufloat(0, 0)
    if event_type == 1 or event_type == 3:
        dBLL = parent.mass - invariant_mass
    if event_type == 2:
        dBLL = invariant_mass - parent.mass
    BLL  = ufloat(0, 0)
    if event_type == 1:
        BLL = get_BLL(N[0]) + dBLL
    if event_type == 2:
        BLL = get_BLL(N[1]) + dBLL

    ##  kinematic fitting  #############################################
    # eta0-matrix: input value
    eta0 = np.zeros((n_track*3, 1))
    for i,d in enumerate(daughters):
        eta0[i*3+0,0] = d.p.n
        eta0[i*3+1,0] = d.theta.n
        eta0[i*3+2,0] = d.phi.n

    # constraints derivation matrix
    D = np.zeros((4, n_track*3))
    for i,d in enumerate(daughters):
        for c in range(4):# constraint
            for val in range(3):# val; 0:mom, 1:theta, 2:phi
                D[c][i*3+val] = d.roundH_rounda(c, val)

    # variance-covariance matrix
    V_eta0 = np.zeros((n_track*3, n_track*3))

    for i,d in enumerate(daughters):
        V_eta0[i*3+0,i*3+0] = d.p.s**2
        V_eta0[i*3+1,i*3+1] = d.theta.s**2
        V_eta0[i*3+2,i*3+2] = d.phi.s**2

    # constraints
    d = np.array([
        [total_final_energy.n - parent.mass.n],
        [total_px.n],
        [total_py.n],
        [total_pz.n]
    ])

    # unknown parameter
    E = np.zeros(( 4, 4 ))
    if event_type == 1 or event_type == 3 or event_type == 0:
        E[0,0] = -1
    if event_type == 2:
        #daughters[0] is double candidate
        E[0,0] = daughters[0].mass.n / np.hypot( daughters[0].mass.n, daughters[0].p.n)

    for c in range(4):# constraint
        for v in range(3):# val; 0:mom, 1:theta, 2:phi
            E[c,v+1] = neutral.roundH_rounda(c, v)

    # solve kinematic fit
    V_D, V_z, V_lambda, V_eta, z, lmbda, eta, chisquare =  solve_kinematic_fit(eta0, V_eta0, D, d, E)

    # new vals
    new_daughters = []
    for i in range(n_track):
        new_daughters.append(
                Particle2(eta[i*3+0,0], np.sqrt(V_eta[i*3+0,i*3+0]),
                          eta[i*3+1,0], np.sqrt(V_eta[i*3+1,i*3+1]),
                          eta[i*3+2,0], np.sqrt(V_eta[i*3+2,i*3+2]),
                          daughters[i].mass.n))

    new_neutral = Particle2(neutral.p.n + z[1,0], np.sqrt(V_z[1,1]),
                            neutral.theta.n + z[2,0], np.sqrt(V_z[2,2]),
                            neutral.phi.n + z[3,0], np.sqrt(V_z[3,3]),
                            N[-1]['M'])

    new_total_final_energy = total_energy(new_daughters)  + new_neutral.energy
    new_total_px = total_momentum(new_daughters, "x") + new_neutral.momentum("x")
    new_total_py = total_momentum(new_daughters, "y") + new_neutral.momentum("y")
    new_total_pz = total_momentum(new_daughters, "z") + new_neutral.momentum("z")
    new_total_p = sqrt(new_total_px**2 + new_total_py**2 + new_total_pz**2)
    if np.fabs(new_total_p.n) < 0.001:
        new_total_p = ufloat( 0., np.sqrt(new_total_px.s**2 + new_total_py.s**2 + new_total_pz.s**2) * 1.05)
    new_invariant_mass_sq = new_total_final_energy**2  - new_total_p**2
    new_invariant_mass = sqrt(new_invariant_mass_sq)

    new_err_m = np.sqrt(V_z[0,0])
    if 'Xi-'  in N[0]['N']:
        new_err_m = sqrt(new_err_m**2 + 0.07**2)
    new_dBLL  = ufloat(-1 * z[0,0], new_err_m)
    new_BLL   = ufloat(0, 0)
    if event_type == 1:
        new_BLL = get_BLL(N[0]) + new_dBLL
    if event_type == 2:
        new_BLL = get_BLL(N[1]) + new_dBLL

    ####################################################################
    # output
    mystr  = "{0} -> {1}".format(N[0]['N'], N[1]['N'])
    for i in range(2,n_track+2):
        mystr  += " + {0}".format(N[i]['N'])
    if event_type == 1:
        mystr  += "  : dBLL = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    if event_type == 2:
        mystr  += "  : dBLL - BXi = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    if event_type == 3:
        mystr  += "  : BXi = {0:3.3f} +/- {1:3.3f}".format(new_dBLL.n, new_dBLL.s)
    mystr  += "  ************************\n"

    mystr += get_str_daughter_vals(daughters)

    mystr += "\n"
    mystr += "KEN = {0:9.3f} +/- {1:9.3f}\n".format(neutral.KE.n, neutral.KE.s)
    mystr += "pN = {0:9.3f} +/- {1:9.3f}\n".format(neutral.p.n, neutral.p.s)
    mystr += "  pNx = {0:9.3f} +/- {1:9.3f}\n".format(neutral.px.n, neutral.px.s)
    mystr += "  pNy = {0:9.3f} +/- {1:9.3f}\n".format(neutral.py.n, neutral.py.s)
    mystr += "  pNz = {0:9.3f} +/- {1:9.3f}\n".format(neutral.pz.n, neutral.pz.s)
    mystr += "thetaN = {0:9.3f} +/- {1:9.3f}\n".format(neutral.theta_deg.n, neutral.theta_deg.s)
    mystr += "phiN   = {0:9.3f} +/- {1:9.3f}\n".format(neutral.phi_deg.n, neutral.phi_deg.s)
    mystr += "\n"
    mystr += "     Calculated_Mass = {0:9.3f} +/- {1:9.3f}\n".format(invariant_mass.n, invariant_mass.s)
    mystr += "      Estimated_Mass = {0:9.3f}\n".format(N[0]['M'])
    mystr += "            Mass_gap = {0:9.3f}\n".format(mass_gap.n)
    mystr += "           Total_mom = {0:9.3f} +/- {1:9.3f}\n".format(total_p.n, total_p.s)
    mystr += "             Total_momx = {0:9.3f} +/- {1:9.3f}\n".format(total_px.n, total_px.s)
    mystr += "             Total_momy = {0:9.3f} +/- {1:9.3f}\n".format(total_py.n, total_py.s)
    mystr += "             Total_momz = {0:9.3f} +/- {1:9.3f}\n\n".format(total_pz.n, total_pz.s)
    mystr += "                   Q = {0:9.3f}\n".format(Q)
    mystr += "Total_Kinetic_Energy = {0:9.3f} +/- {1:9.3f}\n".format(total_ke.n, total_ke.s)
    mystr += "                  QQ = {0:9.3f} +/- {1:9.3f}\n".format(fabs(total_ke.n-Q), total_ke.s)
    mystr += "                dBLL = {0:9.3f} +/- {1:9.3f}\n".format(dBLL.n, dBLL.s)
    mystr += "                BLL  = {0:9.3f} +/- {1:9.3f}\n\n".format(BLL.n, BLL.s)
    mystr += "==== Kinematic Fitting ====\n"

    mystr += get_str_daughter_before_after(daughters, new_daughters)

    mystr += "\n"
    mystr += "pN     : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(neutral.p.n, neutral.p.s, new_neutral.p.n, new_neutral.p.s)
    mystr += "thetaN : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(neutral.theta_deg.n, neutral.theta_deg.s, new_neutral.theta_deg.n, new_neutral.theta_deg.s)
    mystr += "phiN   : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(neutral.phi_deg.n, neutral.phi_deg.s, new_neutral.phi_deg.n, new_neutral.phi_deg.s)
    mystr += "\n"
    mystr += "Total_mom  : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_p.n, total_p.s, new_total_p.n, new_total_p.s)
    mystr += "Total_momx : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_px.n, total_px.s, new_total_px.n, new_total_px.s)
    mystr += "Total_momy : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(total_py.n, total_py.s, new_total_py.n, new_total_py.s)
    mystr += "Total_momz : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n\n" \
            .format(total_pz.n, total_pz.s, new_total_pz.n, new_total_pz.s)
    mystr += "dBLL       : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n" \
            .format(dBLL.n, dBLL.s, new_dBLL.n, new_dBLL.s)
    mystr += "BLL        : {0:9.3f} +/- {1:6.3f}  -> {2:9.3f} +/- {3:6.3f}\n\n\n" \
            .format(BLL.n, BLL.s, new_BLL.n, new_BLL.s)

    fallN.write(mystr)

    if N[-1]['A'] >= 2:
        if event_type == 0:
            if mass_gap >= invariant_mass.s * -cut_sig and \
            QQ >= total_ke.s * -cut_sig:
                fposN.write(mystr)
        if event_type == 1 or event_type == 3:
            if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
            QQ + max_dBLL >= total_ke.s * -cut_sig and \
            dBLL > -1 * max_dBLL:
                fposN.write(mystr)
        if event_type == 2:
            if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
            QQ + max_dBLL >= total_ke.s * -cut_sig and \
            dBLL < max_dBLL:
                fposN.write(mystr)
    else:
        if event_type == 0:
            if fabs(mass_gap) <= invariant_mass.s * cut_sig and \
            fabs(QQ) <= total_ke.s*cut_sig:
                fposN.write(mystr)
        if event_type == 1 or event_type == 3:
            if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
            mass_gap - max_dBLL <= invariant_mass.s * cut_sig and \
            QQ + max_dBLL >= total_ke.s * -cut_sig and \
            QQ - max_dBLL <= total_ke.s * cut_sig and \
            fabs(dBLL) < max_dBLL:
                fposN.write(mystr)
        if event_type == 2:
            if mass_gap + max_dBLL >= invariant_mass.s * -cut_sig and \
            mass_gap - max_dBLL <= invariant_mass.s * cut_sig and \
            QQ + max_dBLL >= total_ke.s * -cut_sig and \
            QQ - max_dBLL <= total_ke.s * cut_sig and \
            fabs(dBLL) < max_dBLL:
                fposN.write(mystr)


def evaluate_combinations(name, particles, event_type, cand_lists, params, with_neutral=False):
    flag_neutral = "N" if with_neutral else ""
    fall = open('result_{0}_all{1}.txt'.format(name, flag_neutral), 'w')
    fpos = open('result_{0}_possible{1}.txt'.format(name,flag_neutral), 'w')
    with fall, fpos:
        for combi in itertools.product(*cand_lists):
            # A number
            if not mass_number_conservation(combi):
                continue
            # charge
            if not charge_conservation(combi):
                continue
            # strangeness
            if 'Xi- &' in combi[0]['N']:#Xi & nucleus
                if not strangeness_conservation(combi):
                    continue
            else:# decay of DLH or SH
                if not one_strangeness_decay(combi):
                    continue
            # Qvalue
            if event_type != 0:
                if Qvalue(combi) + params['max_dBLL'] < -0.001:
                    continue
            else:
                if Qvalue(combi)  < -0.001:
                    continue
            # kinematic
            print_combination(combi)
            if with_neutral:
                evaluate_kinematics_with_neutral(event_type, particles, combi, params, fall, fpos)
            else:
                evaluate_kinematics(event_type, particles, combi, params, fall, fpos)



def run(input_obj, output_dir, cut_sig = 3.0):

    name = input_obj["Name"]
    initial_state = str(input_obj["InitialState"])
    density = input_obj["Density"]
    fragments = input_obj["Fragments"]
    #missingfragment = input_obj["MissingFragment"]
    #output_dir is not used

    print(input_obj)

    particles = []
    particles.append(Particle(density,0,0,0,0,0,0))

    for val in input_obj["Fragments"]:
        particles.append(Particle(
            val["Range"][0],
            val["Range"][1],
            val["Theta"][0],
            val["Theta"][1],
            val["Phi"][0],
            val["Phi"][1],
            val["Strangeness"]))


    #0:　S=-2系がまったく登場しないとき
    #1:　親がダブルのとき。親のほうにdBLLの不定性を持たせる
    #2:　ダブルが娘として存在するとき。始状態にBXiの、娘のほうにdBLLの不定性を持たせる
    #3:　親がXi&原子核束縛系で、ダブルが娘にいない（要するにツイン）場合。始状態にBXiの不定性を持たせる
    event_type = 0
    if initial_state == "DoubleHyper":
        event_type = 1
    for i in range(1, len(particles)):
        if particles[i].Sflag == -2:
            event_type = 2
    if event_type ==0 and initial_state == "XiAtom":
        event_type = 3

    print(len(particles))
    for p in particles:
        print(p.range, p.theta, p.phi, p.Sflag)
    print('density: {0}, track_num: {1}, event_type: {2}'.format(
                  density,
                  len(fragments),
                  event_type))

    # parameters
    params ={
    "cut_sig": cut_sig,
    "max_dBLL": 20,
    "density": density,
    }

    # cand_lists[i_track][i_nucleus]    
    cand_lists = []
    # cand_lists[0] is initial state
    if initial_state == "XiAtom":
        cand_lists.append(Xi_bound_systems())
    elif initial_state == "DoubleHyper":
        cand_lists.append(double_Lambdas())
    elif initial_state == "SingleHyper":
        cand_lists.append(single_Lambdas())
    # cand_lists[1-] are fragments    
    for i,p in enumerate(particles):
        if i == 0:
            continue
        if p.Sflag == -2:
            cand_lists.append(double_Lambdas())
        elif p.Sflag == -1:
            cand_lists.append(single_Lambdas())
        elif p.Sflag == 0:
            cand_lists.append(non_strangeness_chargeds())

    # without neutral particle
    evaluate_combinations(name, particles, event_type, cand_lists, params)

    # with neutral particle
    cand_lists.append(neutrals())
    with_neutral = True
    evaluate_combinations(name, particles, event_type, cand_lists, params, with_neutral)


if __name__ == '__main__':
    sys.argv.append('NagaraA.txt')
    sys.argv.append('-t')

    import kinema_ims
    kinema_ims.run()
