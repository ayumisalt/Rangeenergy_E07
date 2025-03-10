# -*- coding: utf-8 -*-
import numpy as np
import math
import scipy
from uncertainties import ufloat
from uncertainties.umath import sin, cos, sqrt, fabs, acos, atan2
import nuclide
import time
import sys
import os
import itertools
import rangeenergy as reen
import kinema
import importlib
from importlib import machinery
import copy
import warnings
from kinema_impl import normalize, dot_product, cross_product, get_ufloat, get_ufloats
from kinema_impl import calc_coplanarity, calc_costheta, likeness_same_angle, likeness_back2back, balance_momentum
from kinema_impl import format_json, scalar_momentum, kinematic_fitting, str_states, kinematic_mc, calc_density_error
import json
from enum import Enum
import colorama

class TypeReaction(Enum):
    XiToDouble = 220
    XiToSingle = 210
    XiToTwin = 211
    XiToNoHyper = 200
    SigmaToSingle = 110
    SigmaToNoHyper = 100
    DoubleToSingle = 21
    SingleToNoHyper = 10

def set_initial_reaction(initial_state, particles):
    n_single = 0
    n_double = 0
    for particle in particles:
        if particle["Strangeness"] == -1:n_single += 1
        elif particle["Strangeness"] == -2:n_double += 1
        elif particle["Strangeness"] == 0:pass
        else: raise ValueError("Invalid Strangeness")

    # First must be SingleHyper or DoubleHyper
    if n_single == 0 and n_double == 0: pass
    elif n_single == 1 and n_double == 0:pass
    elif n_single == 2 and n_double == 0:pass
    elif n_single > 2 and n_double == 0: raise ValueError("Number of SingleHyper must be <= 2 and >=0")
    elif n_single == 0 and n_double == 1:pass
    elif n_single == 0 and n_double >= 2: raise ValueError("Number of DoubleHyper must be less than or equal to 1")
    else: raise ValueError("Unkown error")

    if initial_state == "XiAtom":
        if n_single == 0 and n_double == 1:type_reaction = TypeReaction.XiToDouble
        elif n_single == 1 and n_double == 0:type_reaction = TypeReaction.XiToSingle
        elif n_single == 2 and n_double == 0:type_reaction = TypeReaction.XiToTwin
        elif n_single == 0 and n_double == 0: type_reaction = TypeReaction.XiToNoHyper
        else: raise ValueError("InitialState is wrong")
    elif initial_state == "SigmaAtom":
        if n_single == 1 and n_double == 0:type_reaction = TypeReaction.SigmaToSingle
        elif n_single == 0 and n_double == 0:type_reaction = TypeReaction.SigmaToNoHyper
        else: raise ValueError("InitialState is wrong")
    elif initial_state == "DoubleHyper" and n_single == 1 and n_double == 0:
        type_reaction = TypeReaction.DoubleToSingle
    elif initial_state == "SingleHyper" and n_single == 0 and n_double == 0:
        type_reaction = TypeReaction.SingleToNoHyper
    else: raise ValueError("InitialState is wrong")

    return type_reaction

def complete_members(particle):
    if "VX" not in particle:
        range_ = get_ufloat(particle, "Range")
        theta = get_ufloat(particle, "Theta")
        phi = get_ufloat(particle, "Phi")
        theta = theta * ufloat(np.pi, 0) / ufloat(180, 0)
        phi = phi * ufloat(np.pi, 0) / ufloat(180, 0)
        vx = range_ * sin(theta) * cos(phi)
        vy = range_ * sin(theta) * sin(phi)
        vz = range_ * cos(theta)
        particle["VX"] = [vx.n, vx.s]
        particle["VY"] = [vy.n, vy.s]
        particle["VZ"] = [vz.n, vz.s]
        particle["NVX"] = [(sin(theta) * cos(phi)).n, (sin(theta) * cos(phi)).s]
        particle["NVY"] = [(sin(theta) * sin(phi)).n, (sin(theta) * sin(phi)).s]
        particle["NVZ"] = [(cos(theta)).n, (cos(theta)).s]
    elif "Range" not in particle:
        vx = get_ufloat(particle, "VX")
        vy = get_ufloat(particle, "VY")
        vz = get_ufloat(particle, "VZ")
        range_ = (vx ** 2 + vy ** 2 + vz ** 2) ** 0.5
        theta = acos(vz / range_) * ufloat(180, 0) / ufloat(np.pi, 0)
        phi = atan2(vy, vx) * ufloat(180, 0) / ufloat(np.pi, 0)
        particle["Range"] = [range_.n, range_.s]
        particle["Theta"] = [theta.n, theta.s]
        particle["Phi"] = [phi.n, phi.s]
        particle["NVX"] = [vx.n / range_.n, vx.s / range_.n]
        particle["NVY"] = [vy.n / range_.n, vy.s / range_.n]
        particle["NVZ"] = [vz.n / range_.n, vz.s / range_.n]

    particle["ThetaRad"] = [particle["Theta"][0] / 180 * np.pi, particle["Theta"][1] / 180 * np.pi]
    particle["PhiRad"] = [particle["Phi"][0] / 180 * np.pi, particle["Phi"][1] / 180 * np.pi]


def ke2mom(mass, KE):
    if KE < 0.001: raise Exception()
    return sqrt((mass + KE) ** 2 - mass ** 2)


def mom2ke(mass, mom):
    return sqrt(mass ** 2 + mom ** 2) - mass


def complete_nuclide(particle, density, density_error):
    if particle["Strangeness"] == 0:
        nuclei = nuclide.normal
    elif particle["Strangeness"] == -1:
        nuclei = nuclide.single_hyper
    elif particle["Strangeness"] == -2:
        nuclei = nuclide.double_hyper
    else:
        raise

    particle["Nuclei"] = []
    kinds = []
    if "Kind" not in particle:
        pass
    elif isinstance(particle["Kind"], list):
        for kind in particle["Kind"]:
            kinds.append(kind)
    else:
        if particle["Kind"] != "Any":
            kinds.append(particle["Kind"])

    for nucleus in nuclei:
        if nucleus["Z"] == 0:
            continue
        if "MinZ" in particle and nucleus["Z"] < particle["MinZ"]:
            continue
        if "MaxZ" in particle and nucleus["Z"] > particle["MaxZ"]:
            continue
        if len(kinds) != 0 and "Any" not in kinds and nucleus["Name"] not in kinds:
            continue
        if "VeryShortLife" in nucleus and nucleus["VeryShortLife"] is True:
            continue

        range_ = get_ufloat(particle, "Range")
        range_straggling = reen.RangeStragglingFromRange(nucleus['M'], range_.n, nucleus['Z'], density)
        range_c = ufloat(range_.n, sqrt(range_.s ** 2 + range_straggling ** 2))
        ke_c = reen.KEfromRange(nucleus['M'], range_.n, nucleus['Z'], density)
        ke_cu = reen.KEfromRange(nucleus['M'], range_.n, nucleus['Z'], density + density_error)
        ke_cl = reen.KEfromRange(nucleus['M'], range_.n, nucleus['Z'], density - density_error)
        mom_c = ke2mom(nucleus['M'], ke_c)
        mom_cu = ke2mom(nucleus['M'], ke_cu)
        mom_cl = ke2mom(nucleus['M'], ke_cl)

        range_l = range_c.n - range_c.s
        ke_l = reen.KEfromRange(nucleus['M'], range_l, nucleus['Z'], density)
        mom_l = ke2mom(nucleus['M'], ke_l)

        range_r = range_c.n + range_c.s
        ke_r = reen.KEfromRange(nucleus['M'], range_r, nucleus['Z'], density)
        mom_r = ke2mom(nucleus['M'], ke_r)

        nucleus["Range"] = [range_c.n, range_c.s]
        nucleus["KE"] = [ke_c, np.fabs(ke_l - ke_r) / 2.0]
        nucleus["Mom"] = [mom_c, np.fabs(mom_l - mom_r) / 2.0]
        nucleus["KE_DensityErrorU"] = [ke_cu, np.fabs(ke_l - ke_r) / 2.0]
        nucleus["Mom_DensityErrorU"] = [mom_cu, np.fabs(mom_l - mom_r) / 2.0]
        nucleus["KE_DensityErrorL"] = [ke_cl, np.fabs(ke_l - ke_r) / 2.0]
        nucleus["Mom_DensityErrorL"] = [mom_cl, np.fabs(mom_l - mom_r) / 2.0]
        nucleus["Mass"] = [nucleus['M'], 0.0]
        particle["Nuclei"].append(copy.deepcopy(nucleus))


def complete_neutral(particle, state, total_mom, total_mom_xyz):
    particle["NVX"] = [-total_mom_xyz[0].n / total_mom.n, total_mom_xyz[0].s / total_mom.n]
    particle["NVY"] = [-total_mom_xyz[1].n / total_mom.n, total_mom_xyz[1].s / total_mom.n]
    particle["NVZ"] = [-total_mom_xyz[2].n / total_mom.n, total_mom_xyz[2].s / total_mom.n]

    ke = mom2ke(state["M"], total_mom)
    state["Range"] = [float('inf'), 0]
    state["KE"] = [ke.n, ke.s]
    state["Mom"] = [total_mom.n, total_mom.s]
    state["Mass"] = [state['M'], 0.0]


def complete_missing(particle, state, residual_mom):
    mom = residual_mom
    state["Mom2"] = [mom.n, mom.s]
    ke = mom2ke(state["M"], get_ufloat(state, "Mom2"))
    state["KE2"] = [ke.n, ke.s]


def complete_missing2(particle, state, mom):
    ke = mom2ke(state["M"], mom)
    state["KE4"] = [ke, 0]


def complete_not_missing(state):
    state["Mom2"] = state["Mom"]
    state["KE2"] = state["KE"]


def complete_fitted(particle, state):
    mom = get_ufloat(particle, "Momentum")
    state["Mom3"] = [mom.n, mom.s]
    ke = mom2ke(state["M"], get_ufloat(state, "Mom3"))
    state["KE3"] = [ke.n, ke.s]


def complete_neutral_fitted(state, total_mom):
    state["Mom3"] = [total_mom.n, 0]
    ke = mom2ke(state["M"], get_ufloat(state, "Mom3"))
    state["KE3"] = [ke.n, ke.s]

def complete_null_fitted(state):
    state["Mom3"] = [0.0, 0.0]
    state["KE3"] = [0.0, 0.0]


def complete_null_neutral(particle, state):
    particle["NVX"] = [(1.0 / 3.0) ** 0.5, 0.0]
    particle["NVY"] = [(1.0 / 3.0) ** 0.5, 0.0]
    particle["NVZ"] = [(1.0 / 3.0) ** 0.5, 0.0]

    state["Range"] = [0, 0]
    state["KE"] = [0, 0]
    state["Mom"] = [0, 0]
    state["Mass"] = [state['M'], 0.0]


def check_key_conservation(states, key):
    buf = 0
    for state in states[1:]:
        buf += state[key]
    return buf - states[0][key]


def check_nucleon_conservation(states, initial_state):
    if check_key_conservation(states, "A") != 0:
        return False
    if check_key_conservation(states, "Z") != 0:
        return False
    if initial_state == "XiAtom" or initial_state == "SigmaAtom":
        if check_key_conservation(states, "S") != 0:
            return False
    else:
        if check_key_conservation(states, "S") != 1:
            return False
    return True


def calc_q_value(states):
    initial_mass = states[0]["M"]
    final_mass = 0.0
    for state in states[1:]:
        final_mass += state["M"]
    return initial_mass - final_mass, final_mass


def calc_total_kinetic_energy(states, name="KE"):
    total_ke = ufloat(0, 0)
    kes = []
    for state in states[1:]:
        total_ke += get_ufloat(state, name)
        kes.append(get_ufloat(state, name))
    return total_ke, kes


def calc_total_energy(states, name="KE"):
    total_energy = ufloat(0, 0)
    for state in states[1:]:
        total_energy += get_ufloat(state, name) + get_ufloat(state, "Mass")
    return total_energy


def calc_total_momentum(particles, states, name="Mom"):
    total_mom_xyz = [ufloat(0, 0), ufloat(0, 0), ufloat(0, 0)]
    moms_xyz = []
    moms = []
    for particle, state in zip(particles[1:], states[1:]):
        mom = get_ufloat(state, name)
        moms.append(mom)
        moms_xyz.append([mom * get_ufloat(particle, "NVX"),
            mom * get_ufloat(particle, "NVY"),
            mom * get_ufloat(particle, "NVZ")])
        total_mom_xyz[0] += moms_xyz[-1][0]
        total_mom_xyz[1] += moms_xyz[-1][1]
        total_mom_xyz[2] += moms_xyz[-1][2]
    total_mom = scalar_momentum(total_mom_xyz)
    return total_mom, total_mom_xyz, moms, moms_xyz


def calc_residual_momentum(moms_xyz, missing_fragment):
    known_mom_xyz = [ufloat(0, 0), ufloat(0, 0), ufloat(0, 0)]
    for i in range(len(moms_xyz)):
        if i != missing_fragment:
            known_mom_xyz[0] += moms_xyz[i][0]
            known_mom_xyz[1] += moms_xyz[i][1]
            known_mom_xyz[2] += moms_xyz[i][2]
    known_mom = scalar_momentum(known_mom_xyz)
    known_vector = {}
    known_vector["NVX"] = [known_mom_xyz[0].n / known_mom.n,known_mom_xyz[0].s / known_mom.n]
    known_vector["NVY"] = [known_mom_xyz[1].n / known_mom.n,known_mom_xyz[1].s / known_mom.n]
    known_vector["NVZ"] = [known_mom_xyz[2].n / known_mom.n,known_mom_xyz[2].s / known_mom.n]
    return known_mom, known_mom_xyz, known_vector


def calc_BL(normal, single_hyper, double_hyper, neutrals):
    for hyper in neutrals:
        if hyper["Name"] == "L":
            mass_L = hyper["M"]
            break

    for hyper in single_hyper:
        Z = hyper["Z"]
        A = hyper["A"]
        if Z == 0:
            hyper["BL"] = 0
            continue
        for norm in normal:
            if norm["Z"] == Z and norm["A"] == (A - 1):
                hyper["BL"] = norm["M"] + mass_L - hyper["M"]

    for hyper in double_hyper:
        Z = hyper["Z"]
        A = hyper["A"]
        if Z == 0:
            continue
        for norm in normal:
            if norm["Z"] == Z and norm["A"] == (A - 2):
                hyper["BL"] = (norm["M"] + mass_L * 2 - hyper["M"]) / 2



def get_initial_state(nuclei, initial_state_kind):
    if len(initial_state_kind) == 0:
        return nuclei
    else:
        kinds = []
        for kind in nuclei:
            if kind["Name"] in initial_state_kind:
                kinds.append(kind)
        if len(kinds) == 0:
            raise ValueError(f"{initial_state_kind} is not defined")
        return kinds

def run(input_obj, output_dir, cut_sig=10.0):

    name = input_obj["Name"]
    initial_state = str(input_obj["InitialState"])
    density = input_obj["Density"]
    if "DensityError" in input_obj: density_error = input_obj["DensityError"]
    else:density_error = 0
    fragments = input_obj["Fragments"]
    additional = ""

    if "Additional" in input_obj:
        additional = input_obj["Additional"]

    if additional == "InFlight":
        inflight_fragment = copy.deepcopy(input_obj["InFlightFragment"])
        complete_members(inflight_fragment)

    missing_fragment = None
    if additional == "Missing":
        missing_fragment = int(input_obj["MissingFragment"])

    input_param = {}
    output_param = {}
    output_params = []

    mom_axis_name = ["MomentumX","MomentumY","MomentumZ"]

    # MC
    input_param["MC"] = 0 if "MC" not in input_obj else input_obj["MC"]

    # significance unit is sigma
    input_param["CutSigma"] = cut_sig

    # unit is MeV
    max_dBLL = 20.0
    max_BXi = 20.0
    min_BXi = 5.0
    input_param["Max_dBLL"] = max_dBLL

    type_reaction = set_initial_reaction(initial_state, fragments)

    # unit is MeV
    if type_reaction == TypeReaction.XiToDouble or type_reaction == TypeReaction.DoubleToSingle:
        min_q_value = -20
    else:
        min_q_value = -2.0  # 3sigma error of BL

    print("Name         = {0}".format(name))
    print("InitialState = {0}".format(initial_state))
    print("Additional   = {0}".format(additional))
    print("TypeReaction =",str(type_reaction))
    if additional == "InFlight":
        print("    InFlightFragment = {0}".format(input_obj["InFlightFragment"]))
    elif additional == "Missing":
        print("    MissingFragment = Daughters_{0}".format(missing_fragment + 1))
    print("Density      = {0} +/- {1}".format(density,density_error))
    print("MC simulation= {0}".format(input_param["MC"]))

    with_neutrals = True
    if "WithNeutrals" in input_obj:
        with_neutrals = input_obj["WithNeutrals"]
    if additional == "InFlight":
        with_neutrals = False
    print("WithNeutrals = {0}".format(with_neutrals))

    for i, fragment in enumerate(fragments):
        print("Fragment_{1} = {0}".format(str(fragment).replace(' ',''), i + 1))

    # Output initial information
    input_param["InitialState"] = initial_state
    input_param["AdditionalState"] = additional
    input_param["Type_reaction"] = str(type_reaction)
    if additional == "Missing":
        input_param["MissingParticleNo"] = missing_fragment
    elif additional == "InFlight":
        input_param["InFlightFragment"] = inflight_fragment
    input_param["Name"] = name
    input_param["WithNeutrals"] = with_neutrals
    input_param["MinQValue"] = min_q_value
    input_param["Density"] = density

    # Complete fragments members [Range Theta Phi VX VY VZ NVX NVY NVZ]
    for i, fragment in enumerate(fragments):
        complete_members(fragment)

    nuclide.normal = sorted(nuclide.normal, key=lambda item: item["A"])
    nuclide.neutrals = sorted(nuclide.neutrals, key=lambda item: item["A"])
    nuclide.double_hyper = sorted(nuclide.double_hyper, key=lambda item: item["A"])
    nuclide.single_hyper = sorted(nuclide.single_hyper, key=lambda item: item["A"])
    nuclide.xi_atom = sorted(nuclide.xi_atom, key=lambda item: item["A"])
    nuclide.sigma_atom = sorted(nuclide.sigma_atom, key=lambda item: item["A"])

    # Calculate BL
    calc_BL(nuclide.normal, nuclide.single_hyper, nuclide.double_hyper, nuclide.neutrals)

    particles = []

    # particles[0] is for inital state
    particles.append({})

    for fragment in fragments:
        particles.append(copy.deepcopy(fragment))

    if additional == "InFlight":
        inflight_particle = copy.deepcopy(inflight_fragment)

    # List of particles, biginning is initial state, end is neutral particle
    states_list = []

    # Check kinds of initlal_state
    initial_state_kinds = []
    if "InitialStateKind" in input_obj:
        if isinstance(input_obj["InitialStateKind"], list):
            for kind in input_obj["InitialStateKind"]:
                initial_state_kinds.append(kind)
        else:
            if input_obj["InitialStateKind"] != "Any":
                initial_state_kinds.append(input_obj["InitialStateKind"])

    # Add inital_state to cand_lists
    if initial_state == "XiAtom":
        states_list.append(get_initial_state(nuclide.xi_atom, initial_state_kinds))
    elif initial_state == "SigmaAtom":
        states_list.append(get_initial_state(nuclide.sigma_atom, initial_state_kinds))
    elif initial_state == "DoubleHyper":
        states_list.append(get_initial_state(nuclide.double_hyper, initial_state_kinds))
    elif initial_state == "SingleHyper":
        states_list.append(get_initial_state(nuclide.single_hyper, initial_state_kinds))
    else:
        raise ValueError(f"{initial_state} is not defined")

    # Add particles to cand_lists
    for particle in particles[1:]:
        complete_nuclide(particle, density, density_error)
        if type_reaction == TypeReaction.DoubleToSingle or type_reaction == TypeReaction.SingleToNoHyper:
            pass
        else:
            particle["Nuclei"] = [p for p in particle["Nuclei"] if p["Name"].find("pi") < 0]
        states_list.append(particle["Nuclei"])

    # Check coplanarity and parallelism
    if additional == "InFlight":
        if len(particles[1:]) == 2:
            cop = calc_coplanarity(inflight_fragment, particles[1:])
            output_param["Coplanarity"] = [cop.n, cop.s]
            output_param["Coplanarity_x1000"] = [cop.n * 1000, cop.s * 1000]
    else:
        if len(particles[1:]) == 3:
            cop = calc_coplanarity(particles[1:])
            output_param["Coplanarity"] = [cop.n, cop.s]
            output_param["Coplanarity_x1000"] = [cop.n * 1000, cop.s * 1000]

    if additional != "InFlight":
        if len(particles[1:]) == 2:
            output_param["Likeness_SameAngle"] = likeness_same_angle(particles[1], particles[2])
            output_param["Likeness_Back2Back"] = likeness_back2back(particles[1], particles[2])
    output_param["PossibleReactions"] = []
    output_param["MassGapMC1Sigma"] = []
    output_param["MassGapMC2Sigma"] = []

    # Add neutral particles (including null)
    # **AFTER** coplanar and opening angle calculation
    if with_neutrals == False:
        states_list.append(list(filter(lambda item: item["Name"] == "null", nuclide.neutrals)))
    else:
        if type_reaction == TypeReaction.XiToNoHyper:
            states_list.append([neutral for neutral in nuclide.neutrals if neutral["S"] == -2])
        elif type_reaction == TypeReaction.XiToSingle or type_reaction == TypeReaction.SigmaToNoHyper:
            states_list.append([neutral for neutral in nuclide.neutrals if neutral["S"] == -1])
        else:
            if type_reaction == TypeReaction.DoubleToSingle or type_reaction == TypeReaction.SingleToNoHyper:
                states_list.append([neutral for neutral in nuclide.neutrals if neutral["S"] == 0])
            else:
                states_list.append([neutral for neutral in nuclide.neutrals if neutral["S"] == 0 and neutral["Name"].find("pi") < 0])
    particles.append({})

    n_all = 1
    input_param["StatesList"] = []
    colorama.init()
    for i_states, states in enumerate(states_list):
        states_str = ""
        for state in states:
            states_str+=state["Name"] + ", "
        input_param["StatesList"].append(states_str)
        print("{} : {} : ".format("Parent" if i_states == 0 else f"Daughters_{i_states}",len(states)), colorama.Fore.LIGHTRED_EX + f"[{states_str}]",)
        print(colorama.Style.RESET_ALL,end="")
        n_all *= len(states)
    colorama.deinit()

    input_param["Fragments"] = fragments
    if additional == "InFlight":
        input_param["InFlightFragment"] = inflight_fragment

    # # of combinations
    n_nucleon_cut = 0
    n_q_value_cut = 0
    n_kinematic_cut = [0, 0]

    # All combinations
    for i_states, states in enumerate(itertools.product(*states_list)):
        if (i_states + 1) % 100 == 1:
            print(i_states + 1,"/",n_all, end="\r")
        elif i_states + 1 == n_all:
            print(i_states + 1,"/",n_all)

        # nucleon conservation
        if not check_nucleon_conservation(states, initial_state):
            continue
        else:
            n_nucleon_cut += 1

        # q_value conservation
        q, final_mass = calc_q_value(states)

        if q < min_q_value:
            continue
        else:
            n_q_value_cut += 1

        initial_mass = states[0]["M"]
        mass_error = 0
        for state in states:
            if "M_Error" in state:
                mass_error += state["M_Error"] ** 2
        mass_error = math.sqrt(mass_error)

        # For output members
        each_output_param = {}
        each_output_param["ReactionFormula"] = str_states(states)
        each_output_param["InitialMass"] = initial_mass
        each_output_param["FinalMass"] = final_mass
        each_output_param["Q"] = q
        each_output_param["Total"] = {}
        each_output_param["Particles"] = []
        each_output_param["MassError"] = mass_error

        # Calc momentum W/O neutral
        total_mom, total_mom_xyz, moms, moms_xyz = calc_total_momentum(particles[:-1], states[:-1])

        multi_neutral = False
        neutral_is_null = False
        if states[-1]["SingleParticle"] == False:
            multi_neutral = True
        if states[-1]["Name"] == "null":
            neutral_is_null = True

        if not neutral_is_null:
            each_output_param["WithMultiNeutral"] = multi_neutral
        each_output_param["WithNeutral"] = not neutral_is_null

        # If have neutral, residual mom set to neutral
        if neutral_is_null:
            complete_null_neutral(particles[-1], states[-1])
        else:
            mom_neutral_xyz = [-1.0 * total_mom_xyz[0], -1.0 * total_mom_xyz[1], -1.0 * total_mom_xyz[2]]
            complete_neutral(particles[-1], states[-1], total_mom, total_mom_xyz)
            total_mom, total_mom_xyz, moms, moms_xyz = calc_total_momentum(particles, states)
        each_output_param["MomentumBalance"] = [balance_momentum(total_mom_xyz), 0]

        # Calc KE etc.
        total_ke, kes = calc_total_kinetic_energy(states)
        total_energy = calc_total_energy(states)
        energy_residual = total_ke - q
        invariant_mass = sqrt(total_energy ** 2 - total_mom ** 2)
        mass_gap = initial_mass - invariant_mass

        each_output_param["Total"]["InvariantMass"] = [invariant_mass.n, invariant_mass.s]
        each_output_param["Total"]["MassGap"] = [mass_gap.n, mass_gap.s]
        each_output_param["Total"]["KE"] = [total_ke.n, total_ke.s]
        each_output_param["Total"]["Energy"] = [total_energy.n, total_energy.s]
        each_output_param["Total"]["EnergyResidual"] = [energy_residual.n, energy_residual.s]
        for axis in range(3):
            each_output_param["Total"][mom_axis_name[axis]] = [total_mom_xyz[axis].n, total_mom_xyz[axis].s]
        each_output_param["Total"]["Momentum"] = [total_mom.n, total_mom.s]

        # Set output members
        for i in range(len(moms_xyz)):
            ParticleParam = {}
            ParticleParam["KE"] = [kes[i].n,kes[i].s]
            if neutral_is_null == False and i == (len(moms_xyz) - 1):
                for axis in range(3):
                    ParticleParam[mom_axis_name[axis]] = [moms_xyz[i][axis].n,moms_xyz[i][axis].s]
            sub_mom = scalar_momentum(moms_xyz[i])
            ParticleParam["Momentum"] = [moms[i].n, moms[i].s]
            each_output_param["Particles"].append(ParticleParam)

        if type_reaction == TypeReaction.DoubleToSingle or type_reaction == TypeReaction.SingleToNoHyper:
            each_output_param["BL"] = states[0]["BL"]
            each_output_param["BL_Type"] = states[0]['BL_Type']

        if additional == "InFlight":
            inflight_mom, inflight_mom_xyz, inflight_vector = calc_residual_momentum(moms_xyz, -1)
            missing_param = {}
            missing_param["LikenessSameAngle"] = likeness_same_angle(inflight_particle, inflight_vector)
            each_output_param["Missing"] = missing_param

        elif additional == "Missing" and neutral_is_null:
            residual_mom, residual_mom_xyz, residual_vector = calc_residual_momentum(moms_xyz, missing_fragment)
            missing_param = {}

            buf_vector = moms_xyz[missing_fragment]
            missing_vector = {}
            missing_vector["NVX"] = [buf_vector[0].n / moms[missing_fragment].n,buf_vector[0].s / moms[missing_fragment].n]
            missing_vector["NVY"] = [buf_vector[1].n / moms[missing_fragment].n,buf_vector[1].s / moms[missing_fragment].n]
            missing_vector["NVZ"] = [buf_vector[2].n / moms[missing_fragment].n,buf_vector[2].s / moms[missing_fragment].n]
            missing_param["LikenessBack2Back"] = likeness_back2back(missing_vector, residual_vector)
            missing_param["Residual"] = residual_vector
            missing_param["Residual"]["Momentum"] = [residual_mom.n,residual_mom.s]
            missing_param["Original"] = missing_vector

            for i, particle in enumerate(particles[1:]):
                if i == missing_fragment:
                    complete_missing(particle, states[1:][i], residual_mom)
                else:
                    complete_not_missing(states[1:][i])

            total_ke2, kes2 = calc_total_kinetic_energy(states, "KE2")
            residual_energy2 = total_ke2 - q
            missing_param["Total"] = {}
            missing_param["Total"]["KE"] = [total_ke2.n, total_ke2.s]
            missing_param["Total"]["EnergyResidual"] = [residual_energy2.n, residual_energy2.s]
            each_output_param["Missing"] = missing_param

        output_params.append(each_output_param)

        # Check kinematics
        if additional == "InFlight":
            each_output_param["MassGapLimits"] = [-((mass_gap.s + mass_error) * cut_sig), q]
            if each_output_param["Missing"]["LikenessSameAngle"] > cut_sig:
                each_output_param["CutCondition"] = "NotSameAngle"
                continue
        elif additional == "Missing" and neutral_is_null == True:
            if each_output_param["Missing"]["LikenessBack2Back"] > cut_sig:
                each_output_param["CutCondition"] = "NotBack2Back"
                continue
            each_output_param["ResidualEnergy2Limits"] = [-1 * (20 + residual_energy2.s * 3), (20 + residual_energy2.s * 3)]
            if residual_energy2.n < (each_output_param["ResidualEnergy2Limits"][0]):
                each_output_param["CutCondition"] = "LowResidualEnergy2"
                continue
            if residual_energy2.n > (each_output_param["ResidualEnergy2Limits"][1]):
                each_output_param["CutCondition"] = "HighResidualEnergy2"
                continue
            if each_output_param["Missing"]["Residual"]["Momentum"][0] < each_output_param["Particles"][missing_fragment]["Momentum"][0]:
                each_output_param["CutCondition"] = "LowMissingKE"
                continue
        elif additional == "Missing" and neutral_is_null == False:
            each_output_param["MassGapLimits"] = [-((mass_gap.s + mass_error) * cut_sig), q]
        else:
            # Momentum balance
            # if each_output_param["MomentumBalance"][0] > 3.78:
            if each_output_param["MomentumBalance"][0] > 10:
                each_output_param["CutCondition"] = "UnbalancedMomentum_3.78"
                continue
            if type_reaction == TypeReaction.DoubleToSingle or type_reaction == TypeReaction.XiToDouble:
                each_output_param["MassGapLimits"] = [-(mass_gap.s * cut_sig + max_dBLL), (mass_gap.s * cut_sig + max_dBLL)]
            elif type_reaction == TypeReaction.XiToTwin:
                each_output_param["MassGapLimits"] = [-(mass_gap.s * cut_sig + min_BXi), (mass_gap.s * cut_sig + max_BXi)]
            else:  # single
                each_output_param["MassGapLimits"] = [-((mass_gap.s + mass_error) * cut_sig), ((mass_gap.s + mass_error) * cut_sig)]

        if "MassGapLimits" in each_output_param.keys():
            if multi_neutral == True:
                each_output_param["MassGapLimits"][1] = initial_mass

            if mass_gap < each_output_param["MassGapLimits"][0]:
                each_output_param["CutCondition"] = "LowMassGap"
                continue
            elif mass_gap > each_output_param["MassGapLimits"][1]:
                each_output_param["CutCondition"] = "HighMassGap"
                continue

        #clear all cut condisiton
        each_output_param["CutCondition"] = "None"
        each_output_param["Elements"] = [s["Name"] for s in states]
        if additional == "Missing" and neutral_is_null == True:
            output_param["PossibleReactions"].append(each_output_param["ReactionFormula"] + " {:+.3f} +/- {:.3f}".format(-residual_energy2.n, residual_energy2.s))
        else:
            output_param["PossibleReactions"].append(each_output_param["ReactionFormula"] + " {:+.3f} +/- {:.3f}".format(mass_gap.n, mass_gap.s))

        if additional == "" and multi_neutral == False and len(fragments) >= 2:

            if input_param["MC"] >= 2:
                # Kinematic MC
                values, mass_gaps = kinematic_mc(initial_mass, particles, states,neutral_is_null,events=input_param["MC"])
                output_param["MassGapMC1Sigma"].append(each_output_param["ReactionFormula"] + " {:+.3f} + {:.3f} - {:.3f}".
                                                       format(values[2], values[3] - values[2],values[2] - values[1]))
                output_param["MassGapMC2Sigma"].append(each_output_param["ReactionFormula"] + " {:+.3f} + {:.3f} - {:.3f}".
                                                       format(values[2], values[4] - values[2],values[2] - values[0]))
                import matplotlib.pyplot as plt
                from statistics import mean, median,variance,stdev
                plt.hist(mass_gaps,bins=20,label="Mean:{:.2f}\nStdev:{:.2f}".format(mean(mass_gaps),stdev(mass_gaps)))
                plt.title(each_output_param["ReactionFormula"])
                plt.legend()
                plt.savefig(os.path.join(output_dir,"{}_{}.png".format(name,len(output_param["MassGapMC1Sigma"]))))
                plt.close()

            mass_gap_ul = calc_density_error(initial_mass, particles, states)
            each_output_param["Total"]["MassGap_DensityError"] = [mass_gap_ul[1] - mass_gap.n,mass_gap_ul[0] - mass_gap.n]

            #Parameters for kinematic fit
            masses = [state["M"] for state in states]
            thetarads = [fragment["ThetaRad"] for fragment in fragments]
            phirads = [fragment["PhiRad"] for fragment in fragments]
            momenta = [particle["Momentum"] for particle in each_output_param["Particles"]]
            if neutral_is_null == False:
                particle = each_output_param["Particles"][-1]
                thetarads.append([acos(particle["MomentumZ"][0] / particle["Momentum"][0])])
                phirads.append([atan2(particle["MomentumY"][0], particle["MomentumX"][0])])

            #Kinematic fit
            new_particles, new_total_mom, new_total_mom_xyz ,chi2 = \
                kinematic_fitting(neutral_is_null, len(fragments), type_reaction, \
                masses, thetarads, phirads, momenta, total_energy, total_mom_xyz)

            each_output_param["KinematicFitting"] = {}
            each_output_param["KinematicFitting"]["Particles"] = new_particles
            each_output_param["KinematicFitting"]["Total"] = {}
            each_output_param["KinematicFitting"]["ChiSquare"] = chi2
            for i, new_particle in enumerate(new_particles):
                complete_fitted(new_particle, states[1:][i])
            if neutral_is_null:
                complete_null_fitted(states[-1])
            total_ke3, kes3 = calc_total_kinetic_energy(states, "KE3")
            total_energy3 = calc_total_energy(states, "KE3")
            energy_residual3 = total_ke3 - q
            invariant_mass3 = sqrt(total_energy3 ** 2 - new_total_mom ** 2)
            mass_gap3 = initial_mass - invariant_mass3
            each_output_param["KinematicFitting"]["Total"]["InvariantMass"] = [invariant_mass3.n, invariant_mass3.s]
            each_output_param["KinematicFitting"]["Total"]["MassGap"] = [mass_gap3.n, mass_gap3.s]
            each_output_param["KinematicFitting"]["Total"]["KE"] = [total_ke3.n, total_ke3.s]
            each_output_param["KinematicFitting"]["Total"]["Energy"] = [total_energy3.n, total_energy3.s]
            each_output_param["KinematicFitting"]["Total"]["EnergyResidual"] = [energy_residual3.n, energy_residual3.s]
            for axis in range(3):
                each_output_param["KinematicFitting"]["Total"][mom_axis_name[axis]] = [new_total_mom_xyz[axis].n, new_total_mom_xyz[axis].s]
            each_output_param["KinematicFitting"]["Total"]["Momentum"] = [new_total_mom.n, new_total_mom.s]
        elif additional == "Missing" and neutral_is_null:
            
            #Kinematic fit to estimate momentum
            for i in range(len(particles[1:])):
                states[1:][i]["Mom4"] = states[1:][i]["Mom"]
                states[1:][i]["KE4"] = states[1:][i]["KE"]

            def estimate_xi2(mom):
                states[1:][missing_fragment]["Mom4"] = [mom[0],0]
                complete_missing2(particles[1:][missing_fragment], states[1:][missing_fragment], mom[0])
                total_mom_buf, total_mom_xyz_buf, moms_buf, moms_xyz_buf = calc_total_momentum(particles, states, "Mom4")
                total_energy_buf = calc_total_energy(states, "KE4")
                invariant_mass_buf = sqrt(total_energy_buf ** 2 - total_mom_buf ** 2)
                mass_gap_buf = initial_mass - invariant_mass_buf
                chi2 = 0
                for ax in range(3):
                    chi2+= (total_mom_xyz_buf[ax].n / total_mom_xyz_buf[ax].s) ** 2 
                chi2 += (mass_gap_buf.n / mass_gap_buf.s) ** 2
                if False: #comment
                    print(f"mom={mom[0]:.5f}",end=" ")
                    print("total_mom_xyz=",end="")
                    print([f"{buf.n:.3f}±{buf.s:.3f}"for buf in total_mom_xyz_buf],end="")
                    print(f"mass_gap={mass_gap_buf.n:.3f}±{mass_gap_buf.s:.3f}",f"chi2={chi2:.3f}")
                return chi2

            best_mom = residual_mom.n
            res = scipy.optimize.minimize(estimate_xi2, best_mom)
            each_output_param["Missing"]["Chi2"] = res["fun"]
            each_output_param["Missing"]["OptimumMomentum"] = best_mom
            

        if neutral_is_null:
            n_kinematic_cut[0] += 1
        else:
            n_kinematic_cut[1] += 1


    combination_param = {}
    combination_param["All"] = n_all
    combination_param["NucleonCut"] = n_nucleon_cut
    combination_param["QValueCut"] = n_q_value_cut
    combination_param["KinematicCut"] = n_kinematic_cut
    output_param["Combination"] = combination_param
    cut_condition = {}
    cut_condition["Type"] = ["WithoutNeutral","WithNeutral","WithMultiNeutral"]
    for par in output_params:
        n_id = 0
        if par["WithNeutral"]:
            if par["WithMultiNeutral"]:
                n_id = 2
            else:
                n_id = 1

        if par["CutCondition"] not in cut_condition:
            cut_condition[par["CutCondition"]] = [0,0,0]
        cut_condition[par["CutCondition"]][n_id]+=1
    output_param["CutCondition"] = cut_condition

    obj = {}
    obj["InputParam"] = input_param
    obj["OutputParam"] = output_param
    obj["OutputParams"] = output_params

    print(f"{n_all}(all) -> {n_nucleon_cut}(nucleon) -> {n_q_value_cut}(q_value) -> {n_kinematic_cut}(kinematic)")

    # Declare output files
    output_json = json.dumps(obj, indent=1)
    output_filename = os.path.join(output_dir, f"{name}_all.json")
    format_json(output_json, output_filename)
    print(f"File {output_filename} output is complete.")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('runcard.json')
    sys.argv.append('-p')
    sys.argv.append('kinema')

    import kinema_ims
    kinema_ims.run()
