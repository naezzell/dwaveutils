# import dependencies
import numpy as np
from dwave.system.samplers import DWaveSampler
import itertools
from collections import OrderedDict
import pandas as pd


def find_heff_s(h, eps, chipschedule):
    """
    Strength of transverse field bias term is proportionally to B(s) weight, but also is diminshed
    by its relative strength to A(s). Hence, heff = B(s) / A(s). Given a h, this function finds
    the value of s whose heff is h up to a tolerance of eps.
    """
    # reassign chipdata to individual variables for reading
    svals = chipschedule['s']
    Avals = chipschedule['A(s) (GHz)']
    Bvals = chipschedule['B(s) (GHz)']
    heffs = np.array(Avals / Bvals)
    index = np.argmin([abs(x - h) for x in heffs])

    if abs(heffs[index] - h) > eps:
        raise ValueError("No heff exists within tolerance eps of h.")

    return svals[index]

def create_heff_csv(chipdataf, newfile):
    """
    Creates a csv file that maps heff values to s annealing parameter values.
    """
    chipschedule = pd.read_csv(chipdataf)
    svals = chipschedule['s']
    Avals = chipschedule['A(s) (GHz)']
    Bvals = chipschedule['B(s) (GHz)']
    heffs = np.array(Avals / Bvals)
    
    df = pd.DataFrame({'heff': heffs, 's': svals})
    df.to_csv(newfile, index=False)
    
    return "Great Success."


def make_anneal_schedule(direction, s, ta, tp=0, tq=0):
    """
    Function that creates basic anneal schedules with the following logic:
    anneal in (direction) for (ta) micro seconds until (s) reached,
    pause for (tp) and quench at a slope of mq (default value is max speed)

    Inputs
    ------
    s: float, value of s you wish to anneal to
    ta: float, value of t you wish to anneal to before pausing/quenching
    tp: float, length of pause after annealing for ta
    tq: float, time-length of quench (the faster, the more abrupt, but 1 is max on 2000Q Hardware)

    Output
    ------
    anneal_schedule: a list of (list) tuples of the form [[t0, s0], [t1, s1], ... [sf, tf]]
    """
    # get DWave sampler as set-up in dwave config file and save relevant properites
    sampler = DWaveSampler()
    mint, maxt = sampler.properties["annealing_time_range"]

    # first, ensure that quench slope is within chip bounds
    if tq != 0 and tq < mint:
        raise ValueError("Minimum value of tq possible by chip is: {mint}".format(mint=mint))
    # now, check that anneal time is not too long (quench has equivalent anneal time)
    if (ta + tp + tq) > maxt:
        raise ValueError("Maximum allowed anneal time is: {maxt}.".format(maxt=maxt))
    #make sure s is valid
    if s > 1.0:
        raise ValueError("s cannot exceed 1.")
    
    #if s = 1, stop the anneal after ta micro seconds 
    elif s == 1:
        return [[0, 0], [ta, 1]]
    
    #otherwise, create anneal schedule according to times/ s
    if direction.lower()[0] == 'f':
        sch = [[0, 0], [ta, s], [ta+tp, s], [ta+tp+tq, 1]]
    elif direction.lower()[0] == 'r':
        sch = [[0, 1], [ta, s], [ta+tp, s], [ta+tp+tq, 1]]
    
    #remove duplicates while preserving order (for example if tp = 0)
    ann_sch = list(map(list, OrderedDict.fromkeys(map(tuple, sch))))
    
    return ann_sch


def fixed_relationship_sweep(input_params, together):
    """
    Inputs
    ------
    input_params: {qorc1:[x1, x2], qorc2:[x3, x4], qorc3:[y1, y2], qorc4:[y3, y4]]}
    dictionary mapping qubits or couplers to parameter lists to iterate through
    together: [[qorc1, qorc2], [qorc3, qorc4]]
    list of qubit lists that specify which qubit parameters to sweep with a fixed relationship

    Output
    ------
    fixed_rel_sweep: [{trial1}, {trial2}, ...{trialn}] where qubits labelled as "together" are
    swept with fixed 1-1 relationship, ie, above produces:
    [{qorc1:x1, qorc2:x3, qorc3:y1, qorc4:y3}, {qorc1:x1, qorc2:x3, qorc3:y2, qorc4:y4},
    {qorc1:x2, qorc2:x4, qorc3:y1, qorc4:y3},{qorc1:x2, qorc2:x4, qorc3:y2, qorc4:y4}]
    """
    # list of qubits or couplers
    qsorcs = []
    # index representation of params, as cartesian product must respect fixed positions
    # of arguments and not their values, ie [x1, x3] vary together in example
    idxrep = {}
    for key, value in input_params.items():
        qsorcs.append(key)
        idxrep[key] = [i for i in range(len(value))]

    # remove redundancy in index representation governed by fixed relationships in together
    for fix_rel in together:
        for j in range(len(fix_rel)):
            if j != 0:
                del idxrep[fix_rel[j]]

    # sweep combinations via cartesian product
    idxdict_combos = (list(dict(zip(idxrep, x)) for x in itertools.product(*idxrep.values())))

    # reconstruct actual parameter combinations with "redundant" parameter values
    dict_combos = []
    for combo in idxdict_combos:
        # add back in "redundant" parameters
        for fix_rel in together:
            for qorc in fix_rel[1::]:
                combo[qorc] = combo[fix_rel[0]]

        # add back in true values corresponding to indices
        tempdict = {}
        for key, value in combo.items():
            tempdict[key] = input_params[key][value]

        dict_combos.append(tempdict)

    return dict_combos


def pfm(states):
    """
    Function that determines probabilility that states are ferromagnetic (all up or all down)

    Input
    -----
    states = {(1, 1, 1): 5} means state (1, 1, 1) was measured 5 times
    """
    # total number of samples
    partition = sum(states.values())
    # get number of qubits specified in a state
    qnum = len(next(iter(states.keys())))
    # get number of states that are all up or all down
    up = tuple(1 for i in range(qnum))
    down = tuple(-1 for i in range(qnum))

    if up not in states:
        p_up = 0
    else:
        p_up = states[up]

    if down not in states:
        p_down = 0
    else:
        p_down = states[down]

    pfm = (p_up + p_down) / partition

    return pfm


def get_dwaveH(H, vartype):
    """
    Takes as input an H with problem encoded as dict, graph, etc. and converts to D-Wave readable H
    """

    if isinstance(H, dict):
        return dictH_to_dwaveH(H, vartype)


def dictH_to_dwaveH(H, vartype):
    """
    Converts dict H to D-Wave H as QUBO or Ising.
    """
    if vartype == 'ising':
        h = {}
        J = {}
        for key, value in H.items():
            if key[0] == key[1]:
                h[key[0]] = value
            else:
                J[key] = value
        return [h, J]

    elif vartype == 'qubo':
        return H
