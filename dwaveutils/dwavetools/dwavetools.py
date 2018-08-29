# import dependencies
import numpy as np
from functools import reduce
from dwave.system.samplers import DWaveSampler
import itertools
from collections import OrderedDict
import pandas as pd
import qutip as qt
import qutip.states as qts
import qutip.operators as qto


def make_numeric_schedule(discretization, **kwargs):
    """
    Creates an anneal_schdule to be used for numerical calculatins with QuTip. 
    Returns times and svals associated with each time that [times, svals]
    that together define an anneal schedule.
    
    Inputs: 
    discretization: determines what step size to use between points
    kwargs: dictionary that contains optional key-value args
    optional args: sa, ta, tp, tq
        sa: s value to anneal to 
        ta: time it takes to anneal to sa
        tp: time to pause after reaching sa
        tq: time to quench from sa to s = 1
    """
    # Parse the kwargs input which encodes anneal schedule parameters
    try:
        ta = kwargs['ta']
    except KeyError:
        raise KeyError("An anneal schedule must at least include an anneal time, 'ta'")
    
    # extracts anneal parameter if present; otherwise, returns an empty string
    sa = kwargs.get('sa', '')
    tp = kwargs.get('tp', '')
    tq = kwargs.get('tq', '')
    
    # turn discretization into samplerate multiplier
    samplerate = 1 / discretization
    
    # if no sa present, create a standard forward anneal for ta seconds
    if not sa:
        # determine slope of anneal
        sa = 1; ta = kwargs['ta'];
        ma = sa / ta
        
        # create a list of times with (ta+1)*samplerate elements
        t = np.linspace(0, ta, int((ta+1)*samplerate))
        
        # create linear s(t) function
        sfunc = ma*t
        
    # if no pause present, anneal forward for ta to sa then quench for tq to s=1
    elif not tp:
        # determine slopes and y-intercept (bq) to create piece-wise function
        ma = sa / ta
        mq = (1 - sa) / tq
        bq = (sa*(ta + tq) - ta)/tq
        
        # create a list of times where sampling for anneal/ quench proportional to time there
        t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                np.linspace(ta+.00001, ta+tq, int((ta+tq+1)*samplerate))))
        
        # create a piece-wise-linear (PWL) s(t) function defined over t values
        sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tq)],
                            [lambda t: ma*t, lambda t: bq + mq*t])
        
    # otherwise, forward anneal, pause, then quench
    else:
        # determine slopes and y-intercept (bq) to create piece-wise function 
        ma = sa / ta
        mp = 0
        mq = (1 - sa) / tq
        bq = (sa*(ta + tp + tq) - (ta + tp))/tq
        
        # create a list of times with samplerate elements from 0 and T = ta + tp + tq
        t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                np.linspace(ta+.00001, ta+tp, int((ta+tp+1)*samplerate)),
                                np.linspace(ta+tp+.00001, ta+tp+tq, int((ta+tp+tq+1)*samplerate))))
        
        # create a piece-wise-linear (PWL) s(t) function defined over t values
        sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tp),(ta+tp < t) & (t <= ta+tp+tq)],
                             [lambda t: ma*t, lambda t: sa, lambda t: bq + mq*t])
        
        
    return [t, sfunc]

def nqubit_1pauli(pauli, i, n):
    """
    Creates a single-qubit pauli operator on qubit i (0-based)
    that acts on n qubits. (padded with identities).
    
    For example, pauli = Z, i = 1 and n = 3 gives:
    Z x I x I, an 8 x 8 matrix
    """ 
    #create identity padding
    iden1 = [qto.identity(2) for j in range(i)]
    iden2 = [qto.identity(2) for j in range(n-i-1)]
    
    #combine into total operator list that is in proper order
    oplist = iden1 + [pauli] + iden2
    
    #create final operator by using tensor product on unpacked operator list
    operator = qt.tensor(*oplist)
    
    return operator

def nqubit_2pauli(ipauli, jpauli, i, j, n):
    """
    Creates a 2 qubit x/y/z pauli operator on qubits i,j
    with i < j that acts on n qubits in total.
    
    For example, ipauli = Y, jpauli = Z, i = 1, j = 2 and n = 3 gives:
    Y x Z x I, an 8 x 8 matrix
    """ 
    #create identity padding
    iden1 = [qto.identity(2) for m in range(i)]
    iden2 = [qto.identity(2) for m in range(j-i-1)]
    iden3 = [qto.identity(2) for m in range(n-j-1)]
    
    #combine into total operator list
    oplist = iden1 + [ipauli] + iden2 + [jpauli] + iden3
    
    # apply tensor product on unpacked oplist
    operator = qt.tensor(*oplist)
    
    return operator

def dict_to_qutip(dictrep, encoded_params=None):
    """
    Takes a DictRep Ising Hamiltonian and converts it to a QuTip Ising Hamiltonian.
    Encoded params must be passed if dictrep weights are variables (abstract) and 
    not actual numbers. 
    """
    # make useful operators
    sigmaz = qto.sigmaz()
    nqbits = len(dictrep.qubits)
    zeros = [qto.qzero(2) for m in range(nqbits)]
    finalH = qt.tensor(*zeros)
    
    for key, value in dictrep.H.items():
        if key[0] == key[1]:
            if encoded_params:
                finalH += encoded_params[value]*nqubit_1pauli(sigmaz, key[0], nqbits)
            else:
                finalH += value*nqubit_1pauli(sigmaz, key[0], nqbits)
        else:
            if encoded_params:
                finalH += encoded_params[value]*nqubit_2pauli(sigmaz, sigmaz, key[0], key[1], nqbits)
            else:
                finalH += value*nqubit_2pauli(sigmaz, sigmaz, key[0], key[1], nqbits)
            
    return finalH

def time_interpolation(schedule, processor_data):
    """
    Interpolates the A(s) and B(s) functions in terms of time in accordance with an 
    annealing schedule s(t). Returns cubic-splines amenable to use with QuTip. 
    """
    
    svals = processor_data['svals']
    Avals = processor_data['Avals']
    Bvals = processor_data['Bvals']
    
    # interpolate Avals and Bvals into a cubic spline function
    Afunc = qt.interpolate.Cubic_Spline(svals[0], svals[-1], Avals)
    Bfunc = qt.interpolate.Cubic_Spline(svals[0], svals[-1], Bvals)
        
    # now, extract s(t)
    times = schedule[0]
    sprogression = schedule[1]
    
    # interpolate A/B funcs with respect to time with s(t) relationship implicitly carried through
    sch_Afunc = qt.interpolate.Cubic_Spline(times[0], times[-1], Afunc(sprogression))
    sch_Bfunc = qt.interpolate.Cubic_Spline(times[0], times[-1], Bfunc(sprogression))
    
    sch_ABfuncs = {'A(t)': sch_Afunc, 'B(t)': sch_Bfunc}
    
    return sch_ABfuncs

def loadAandB(file="processor_annealing_schedule_DW_2000Q_2_June2018.csv"):
    """
    Loads in A(s) and B(s) data from chip and interpolates using QuTip's
    cubic-spline function. Useful for numerical simulations.
    
    Returns (as list in this order):
    svals: numpy array of discrete s values for which A(s)/B(s) are defined
    Afunc: interpolated A(s) function
    Bfunc: interpolated B(s) function
    """
    
    Hdata = pd.read_csv(file)
    # pd as in pandas Series form of data
    pdA = Hdata['A(s) (GHz)']
    pdB = Hdata['B(s) (GHz)']
    pds = Hdata['s'] 
    Avals = np.array(pdA)
    Bvals = np.array(pdB)
    svals = np.array(pds)
    
    processor_data = {'svals': svals, 'Avals': Avals, 'Bvals': Bvals}
    
    return processor_data

def get_numeric_H(dictrep):
    HZ = dict_to_qutip(dictrep)
    nqbits = len(dictrep.qubits)
    HX = sum([nqubit_1pauli(qto.sigmax(), m, nqbits) for m in range(nqbits)])
    
    H = {'HZ': HZ, 'HX': HX}
    
    return H


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


def make_dwave_schedule(direction, s, ta, tp=0, tq=0):
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
    #if tq != 0 and tq < mint:
    #    raise ValueError("Minimum value of tq possible by chip is: {mint}".format(mint=mint))
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
