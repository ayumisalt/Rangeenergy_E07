# -*- coding: utf-8 -*-
from uncertainties import ufloat
from uncertainties.umath import sin, cos, sqrt, fabs, acos, atan2
import math
import re
import json
import numpy as np
import ims
import kinema
import rangeenergy as reen
from scipy.optimize import fsolve

def normalize(v0):
    return sqrt(v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2)


def dot_product(v0, v1):
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]


def cross_product(v0, v1):
    vcross = [0, 0, 0]
    vcross[0] = v0[1] * v1[2] - v0[2] * v1[1]
    vcross[1] = v0[2] * v1[0] - v0[0] * v1[2]
    vcross[2] = v0[0] * v1[1] - v0[1] * v1[0]
    return vcross

# @input v[ufloat_range, ufloat_theta, ufloat_phi]
# @output ufloat_cross_dot_product
def dot_cross_product(v0, v1, v2):

    px0 = sin(v0[1]) * cos(v0[2])
    py0 = sin(v0[1]) * sin(v0[2])
    pz0 = cos(v0[1])
    px1 = sin(v1[1]) * cos(v1[2])
    py1 = sin(v1[1]) * sin(v1[2])
    pz1 = cos(v1[1])
    px2 = sin(v2[1]) * cos(v2[2])
    py2 = sin(v2[1]) * sin(v2[2])
    pz2 = cos(v2[1])

    cop = px0 * py1 * pz2 + px1 * py2 * pz0 + px2 * py0 * pz1 + \
        -(px0 * py2 * pz1 + px1 * py0 * pz2 + px2 * py1 * pz0)

    dcop_dtheta0 = (cos(v0[1]) * cos(v0[2])) * py1 * pz2 + \
        +px1 * py2 * (-1 * sin(v0[1])) + \
        +px2 * (cos(v0[1]) * sin(v0[2])) * pz1 + \
        -(cos(v0[1]) * cos(v0[2])) * py2 * pz1 + \
        -px1 * (cos(v0[1]) * sin(v0[2])) * pz2 + \
        -px2 * py1 * (-1 * sin(v0[1]))

    dcop_dphi0 = (-1 * sin(v0[1]) * sin(v0[2])) * py1 * pz2 + \
        +0 + \
        +px2 * (sin(v0[1]) * cos(v0[2])) * pz1 + \
        -(-1 * sin(v0[1]) * sin(v0[2])) * py2 * pz1 + \
        -px1 * (sin(v0[1]) * cos(v0[2])) * pz2 + \
        -0

    dcop_dtheta1 = (cos(v1[1]) * cos(v1[2])) * py2 * pz0 + \
        +px2 * py0 * (-1 * sin(v1[1])) + \
        +px0 * (cos(v1[1]) * sin(v1[2])) * pz2 + \
        -(cos(v1[1]) * cos(v1[2])) * py0 * pz2 + \
        -px2 * (cos(v1[1]) * sin(v1[2])) * pz0 + \
        -px0 * py2 * (-1 * sin(v1[1]))

    dcop_dphi1 = (-1 * sin(v1[1]) * sin(v1[2])) * py2 * pz0 + \
        +0 + \
        +px0 * (sin(v1[1]) * cos(v1[2])) * pz2 + \
        -(-1 * sin(v1[1]) * sin(v1[2])) * py0 * pz2 + \
        -px2 * (sin(v1[1]) * cos(v1[2])) * pz0 + \
        -0

    dcop_dtheta2 = (cos(v2[1]) * cos(v2[2])) * py0 * pz1 + \
        +px0 * py1 * (-1 * sin(v2[1])) + \
        +px1 * (cos(v2[1]) * sin(v2[2])) * pz0 + \
        -(cos(v2[1]) * cos(v2[2])) * py1 * pz0 + \
        -px0 * (cos(v2[1]) * sin(v2[2])) * pz1 + \
        -px1 * py0 * (-1 * sin(v2[1]))

    dcop_dphi2 = (-1 * sin(v2[1]) * sin(v2[2])) * py0 * pz1 + \
        +0 + \
        +px1 * (sin(v2[1]) * cos(v2[2])) * pz0 + \
        -(-1 * sin(v2[1]) * sin(v2[2])) * py1 * pz0 + \
        -px0 * (sin(v2[1]) * cos(v2[2])) * pz1 + \
        -0

    cop_error = sqrt(pow(dcop_dtheta0 * v0[1].s, 2.) + \
        pow(dcop_dphi0 * v0[2].s, 2.) + \
        pow(dcop_dtheta1 * v1[1].s, 2.) + \
        pow(dcop_dphi1 * v1[2].s, 2.) + \
        pow(dcop_dtheta2 * v2[1].s, 2.) + \
        pow(dcop_dphi2 * v2[2].s, 2.))
    #print("cop.s, cop_error = {0}, {1}".format(cop.s, cop_error.n))
    return ufloat(cop.n, cop_error.n)


def dot_cross_product_unusable(v0, v1, v2):
    px0 = sin(v0[1]) * cos(v0[2])
    py0 = sin(v0[1]) * sin(v0[2])
    pz0 = cos(v0[1])
    px1 = sin(v1[1]) * cos(v1[2])
    py1 = sin(v1[1]) * sin(v1[2])
    pz1 = cos(v1[1])
    px2 = sin(v2[1]) * cos(v2[2])
    py2 = sin(v2[1]) * sin(v2[2])
    pz2 = cos(v2[1])
    product = px0 * py1 * pz2 + px1 * py2 * pz0 + px2 * py0 * pz1 - \
        (px0 * py2 * pz1 + px1 * py0 * pz2 + px2 * py1 * pz0)
    return product


def get_ufloat(particle, member):
    return ufloat(particle[member][0], std_dev=particle[member][1])


def get_ufloats(particle, member0, member1, member2):
    return [ufloat(particle[member0][0], std_dev=particle[member0][1]),
            ufloat(particle[member1][0], std_dev=particle[member1][1]),
            ufloat(particle[member2][0], std_dev=particle[member2][1])]

def calc_coplanarity(*args):
    '''
    Parameters
        missingfragment and particles (with 2 particles) = calc_coplanarity(missingfragment, particles)
        or
        particles (with 3 particles) = calc_coplanarity(particles)
    '''
    if len(args) == 2:
        missingfragment = args[0]
        particles = args[1]
        vectors = []
        vectors.append(get_ufloats(missingfragment, "Range", "ThetaRad", "PhiRad"))
        vectors.append(get_ufloats(particles[0], "Range", "ThetaRad", "PhiRad"))
        vectors.append(get_ufloats(particles[1], "Range", "ThetaRad", "PhiRad"))
        return dot_cross_product(vectors[0], vectors[1], vectors[2])
    elif len(args) == 1:
        particles = args[0]
        vectors = []
        vectors.append(get_ufloats(particles[0], "Range", "ThetaRad", "PhiRad"))
        vectors.append(get_ufloats(particles[1], "Range", "ThetaRad", "PhiRad"))
        vectors.append(get_ufloats(particles[2], "Range", "ThetaRad", "PhiRad"))
        return dot_cross_product(vectors[0], vectors[1], vectors[2])
    else:
        raise

def calc_costheta(particles):
    v0 = get_ufloats(particles[0], "NVX", "NVY", "NVZ")
    v1 = get_ufloats(particles[1], "NVX", "NVY", "NVZ")
    ret = dot_product(v0, v1)
    if ret.n >= 1:
        ret = ufloat(1, ret.s)
    elif ret.n <= -1:
        ret = ufloat(-1, ret.s)
    return ret

def distance_from_center(vec):
    n = 0.0
    for v in vec:
        n += v.n ** 2
    if n < 0.0001 ** 2:
        return 0
    for v in vec:
        if math.fabs(v.s) < 0.0001:
            return 10000
    n = 0.0
    for v in vec:
        n += v.n ** 2 / v.s ** 2
    return math.sqrt(n)

def likeness_same_angle(p1, p2):
    vec = []
    vec.append(get_ufloat(p1,"NVX") - get_ufloat(p2,"NVX"))
    vec.append(get_ufloat(p1,"NVY") - get_ufloat(p2,"NVY"))
    vec.append(get_ufloat(p1,"NVZ") - get_ufloat(p2,"NVZ"))
    return distance_from_center(vec)

def likeness_back2back(p1, p2):
    vec = []
    vec.append(get_ufloat(p1,"NVX") + get_ufloat(p2,"NVX"))
    vec.append(get_ufloat(p1,"NVY") + get_ufloat(p2,"NVY"))
    vec.append(get_ufloat(p1,"NVZ") + get_ufloat(p2,"NVZ"))
    return distance_from_center(vec)

def balance_momentum(total_mom_xyz):
    return distance_from_center(total_mom_xyz)


def scalar_momentum(total_mom_xyz):
    total_mom = sqrt(total_mom_xyz[0] ** 2 + total_mom_xyz[1] ** 2 + total_mom_xyz[2] ** 2)
    if str(total_mom.s) == str(float('nan')):
        total_mom = ufloat(total_mom.n, sqrt(total_mom_xyz[0].s ** 2 + total_mom_xyz[1].s ** 2 + total_mom_xyz[2].s ** 2))
    return  total_mom

def format_json(input_text, output_name):
    re_float = r'[+-]?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?'
    pattern0 = r'\[(\s+) [+.eE0-9-]+,\1 [+.eE0-9-]+\1\]'
    pattern1 = r'(' + re_float + '),\s+(' + re_float + ')'
    prog0 = re.compile(pattern0)
    prog1 = re.compile(pattern1)

    pattern2 = r': [+-]?[0-9]+(?:\.[0-9]*)(?:[eE][+-]?[0-9]+)?'
    pattern3 = r': (' + re_float + ')'
    prog2 = re.compile(pattern2)
    prog3 = re.compile(pattern3)

    # input
    searching_strings = input_text

    # split at ReactionFormula
    searching_string_arr = searching_strings.split('\"ReactionFormula\"')
    for i in range(len(searching_string_arr) - 1):
        searching_string_arr[i] += '\"ReactionFormula\"'

    total_string = ""
    for searching_string in searching_string_arr:
        # format 0 -> 1
        sub_string = ""
        m = prog0.search(searching_string)
        while(m):
            a = prog1.findall(m.group(0))
            newstr = '[{0:.3f}, {1:.3f}]'.format(float(a[0][0]),float(a[0][1]))
            sub_string += searching_string[0:m.span()[0]] + newstr
            searching_string = searching_string[m.span()[1]:]
            m = prog0.search(searching_string)
        searching_string = sub_string + searching_string

        # format 2 -> 3
        sub_string = ""
        m = prog2.search(searching_string)
        while(m):
            a = prog3.findall(m.group(0))
            newstr = ': {0:.3f}'.format(float(a[0]))
            sub_string += searching_string[0:m.span()[0]] + newstr
            searching_string = searching_string[m.span()[1]:]
            m = prog2.search(searching_string)
        searching_string = sub_string + searching_string
        total_string += searching_string

    # output
    with open(output_name, 'w') as file:
        file.write(total_string.replace('\r', ''))

def roundH_rounda(i_constraint, val, mass, thetarad, phirad, momentum):
    if i_constraint == 0 and val == 0:
        return momentum / np.sqrt(mass ** 2 + momentum ** 2)
    elif i_constraint == 1 and val == 0:
        return np.sin(thetarad) * np.cos(phirad)
    elif i_constraint == 2 and val == 0:
        return np.sin(thetarad) * np.sin(phirad)
    elif i_constraint == 3 and val == 0:
        return np.cos(thetarad)
    elif i_constraint == 0 and val == 1:
        return 0.0
    elif i_constraint == 1 and val == 1:
        return momentum * np.cos(thetarad) * np.cos(phirad)
    elif i_constraint == 2 and val == 1:
        return momentum * np.cos(thetarad) * np.sin(phirad)
    elif i_constraint == 3 and val == 1:
        return momentum * -np.sin(thetarad)
    elif i_constraint == 0 and val == 2:
        return 0.0
    elif i_constraint == 1 and val == 2:
        return momentum * np.sin(thetarad) * -np.sin(phirad)
    elif i_constraint == 2 and val == 2:
        return momentum * np.sin(thetarad) * np.cos(phirad)
    elif i_constraint == 3 and val == 2:
        return 0.0
    else:
        raise Exception("invalid constraint index")

def calc_fitted_momentum(particles):
    total_mom_xyz = [ufloat(0, 0), ufloat(0, 0), ufloat(0, 0)]
    for i in range(len(particles)):
        mom = get_ufloat(particles[i], "Momentum")
        theta = get_ufloat(particles[i], "Theta") / 180 * np.pi
        phi = get_ufloat(particles[i], "Phi") / 180 * np.pi
        total_mom_xyz[0] += mom * sin(theta) * cos(phi) 
        total_mom_xyz[1] += mom * sin(theta) * sin(phi)
        total_mom_xyz[2] += mom * cos(theta)
    total_mom = scalar_momentum(total_mom_xyz)
    return total_mom, total_mom_xyz

def calc_density_error(initial_mass, particles, states):
    mass_gap_ul = []
    for suffix in ["_DensityErrorU","_DensityErrorL"]:
        for state in states[1:]:
            if "KE" + suffix not in state:state["KE" + suffix] = state["KE"]
            if "Mom" + suffix not in state:state["Mom" + suffix] = state["Mom"]
        total_ke, kes = kinema.calc_total_kinetic_energy(states,"KE" + suffix)
        total_energy = kinema.calc_total_energy(states,"KE" + suffix)
        total_mom, total_mom_xyz, moms, moms_xyz = kinema.calc_total_momentum(particles, states,"Mom" + suffix)
        invariant_mass = sqrt(total_energy ** 2 - total_mom ** 2)
        mass_gap = initial_mass - invariant_mass
        mass_gap_ul.append(mass_gap.n)
    return mass_gap_ul[0], mass_gap_ul[1]

def kinematic_mc(initial_mass, particles, states,neutral_is_null,*,events=1000):
    import copy
    import numpy as np
    rng = np.random.RandomState(123)
    mass_gaps = []
    for i in range(events):
        states2 = copy.deepcopy(states)
        particles2 = copy.deepcopy(particles)
        
        total_mom = ufloat(0,0)
        mom_xyz = [ufloat(0,0),ufloat(0,0),ufloat(0,0)]
        for particle, state in zip(particles2[1:-1],states2[1:-1]):
            ke = rng.normal(state["KE"][0],state["KE"][1])
            mom = kinema.ke2mom(state["M"],ke)
            nx = rng.normal(particle["NVX"][0],particle["NVX"][1])
            ny = rng.normal(particle["NVY"][0],particle["NVY"][1])
            nz = rng.normal(particle["NVZ"][0],particle["NVZ"][1])

            state["KE"] = [ke,0]
            state["Mom"] = [mom,0]
            particle["NVX"] = [nx,0]
            particle["NVY"] = [ny,0]
            particle["NVZ"] = [nz,0]
            
        if neutral_is_null == False:
            total_mom, total_mom_xyz, moms, moms_xyz = kinema.calc_total_momentum(particles2[:-1], states2[:-1])
            kinema.complete_neutral(particles2[-1],states2[-1],total_mom,total_mom_xyz)

        total_ke, kes = kinema.calc_total_kinetic_energy(states2) #states[1:] "KE"
        total_energy = kinema.calc_total_energy(states2)#states[1:] "KE" "Mass"
        total_mom, total_mom_xyz, moms, moms_xyz = kinema.calc_total_momentum(particles2, states2) #states[1:] Mom NVX NVY NVZ
        invariant_mass = sqrt(total_energy ** 2 - total_mom ** 2)
        mass_gap = initial_mass - invariant_mass
        mass_gaps.append(mass_gap.n)

    return (np.percentile(mass_gaps,2.275),
          np.percentile(mass_gaps,15.866),
          np.percentile(mass_gaps,50),
          np.percentile(mass_gaps,100 - 15.866),
          np.percentile(mass_gaps,100 - 2.275)),mass_gaps

def kinematic_fitting(neutral_is_null, n_fragments, type_reaction, 
                      masses, thetarads, phirads, momenta, total_energy, total_mom_xyz):
    """
    概念の話は https://gitlab.com/gifuescan/kinema/wikis/Kinematic-fit を参照せよ
    ここでは引数の説明をする
    type_reaction == XiToDouble のときは、Double生成時の取扱とする
    masses[0]は親核、float
    thetarads, phirads,momentaは娘核と中性粒子のみ[0]と[1]
    total_energyは.nのみ使用
    total_mom_xyzは[0:2].nのみ使用
    """    
    # eta0-matrix: input value
    eta0 = np.zeros((n_fragments * 3, 1))
    for i in range(n_fragments):
        eta0[i * 3 + 0,0] = momenta[i][0]
        eta0[i * 3 + 1,0] = thetarads[i][0]
        eta0[i * 3 + 2,0] = phirads[i][0]

    # constraints derivation matrix
    D = np.zeros((4, n_fragments * 3))
    for i in range(n_fragments):
        for c in range(4):# constraint
            for val in range(3):# val; 0:mom, 1:theta, 2:phi
                D[c][i * 3 + val] = roundH_rounda(c, val, masses[i + 1], thetarads[i][0], phirads[i][0], momenta[i][0])

    # variance-covariance matrix
    V_eta0 = np.zeros((n_fragments * 3, n_fragments * 3))

    for i in range(n_fragments):
        V_eta0[i * 3 + 0,i * 3 + 0] = momenta[i][1] ** 2
        V_eta0[i * 3 + 1,i * 3 + 1] = thetarads[i][1] ** 2
        V_eta0[i * 3 + 2,i * 3 + 2] = phirads[i][1] ** 2

    # constraints
    d = np.array([[total_energy.n - masses[0]],
        [total_mom_xyz[0].n],
        [total_mom_xyz[1].n],
        [total_mom_xyz[2].n]])

    # unknown parameter
    if neutral_is_null:
        E = np.zeros((4, 1))
    else:
        E = np.zeros((4, 4))
        for c in range(4):# constraint
            for v in range(3):# val; 0:mom, 1:theta, 2:phi
                E[c,v + 1] = roundH_rounda(c, v, masses[-1], thetarads[-1][0], phirads[-1][0], momenta[-1][0])

    if type_reaction == kinema.TypeReaction.XiToDouble:
        E[0,0] = masses[1] / np.hypot(masses[1], momenta[-1][0])
    else:
        E[0,0] = -1
    
    V_D, V_z, V_lambda, V_eta, z, lmbda, eta, chi2 = ims.solve_kinematic_fit(eta0, V_eta0, D, d, E)
    
    new_particles = []
    for i in range(n_fragments):
        new_particle = {}
        new_particle["Momentum"] = [eta[i * 3 + 0,0], np.sqrt(V_eta[i * 3 + 0,i * 3 + 0])]
        new_particle["Theta"] = [eta[i * 3 + 1,0] * 180 / np.pi, np.sqrt(V_eta[i * 3 + 1,i * 3 + 1]) * 180 / np.pi]
        new_particle["Phi"] = [eta[i * 3 + 2,0] * 180 / np.pi, np.sqrt(V_eta[i * 3 + 2,i * 3 + 2]) * 180 / np.pi]
        new_particles.append(new_particle)
    new_total_mom, new_total_mom_xyz = calc_fitted_momentum(new_particles)
    if neutral_is_null == False:
        new_particle = {}
        new_particle["Momentum"] = [new_total_mom.n, 0.0]
        new_particle["Theta"] = [acos(-new_total_mom_xyz[2].n / new_total_mom.n) * 180 / np.pi, 0.0]
        new_particle["Phi"] = [atan2(-new_total_mom_xyz[1].n, -new_total_mom_xyz[0].n) * 180 / np.pi, 0.0]
        new_particles.append(new_particle)
        new_total_mom, new_total_mom_xyz = calc_fitted_momentum(new_particles)
    return new_particles, new_total_mom, new_total_mom_xyz , chi2


def str_states(states):
    states2 = states
    if states[-1]["Name"] == "null":
        states2 = states[:-1]

    st = ""
    st += str(states2[0]["Name"]) + " -> "
    for nucleus in states2[1:-1]:
        st += str(nucleus["Name"]) + " + "
    st += str(states2[-1]["Name"])
    return st

def calc_mom(mass1, mass2, q):
    func = lambda mom : math.sqrt(mass1 ** 2 + mom ** 2) + math.sqrt(mass2 ** 2 + mom ** 2) - (mass1 + mass2 + q)
    solution = fsolve(func, 0)
    mom = solution[0]
    return mom