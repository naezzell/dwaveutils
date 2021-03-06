# import dependencies
import numpy as np
from functools import reduce
from dwave.system.samplers import DWaveSampler
import itertools
from collections import OrderedDict
from scipy.stats import entropy as entropy
import pandas as pd
import qutip as qt
import qutip.states as qts
import qutip.operators as qto
from operator import add
import networkx as nx
from math import floor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import random

def densify_unitcell(fqbit, hweight, Jweight, wqubits, wcouplers, ucsize = 8):
    '''
    Heuristically makes the most dense connection motif within a unit
    cell with ucsize qubits whose first (lowest #) qubit is fqbit
    
    Inputs
    ------
    fqbit: index of first qubit in unit cell
    hweight: desired linear weight added to qubits in chain
    Jweight: desired coupling strength between between qubits in chain
    wqubits: list of working qubits (i.e. [0, 1, 2, ...])
    wcopulers: list of working couplings (i.e. [[0, 1], [0, 2],...])
    '''
    H = {}
    #print(ucsize - 1)
    lqubit = fqbit + (ucsize-1)
    qubits = []
    couplings = []
    if lqubit in wqubits:
        qubits.append(lqubit)
        
    # begin big heuristic loop
    # flags true when unit cell is heuristically, densely connected
    #complete = False
    #while not complete:
        # find first last qubit that is in working list (adjust later if necessary to end loop)
    #    if lqubit not in wqubits:
    #        lqubit -= 1
    #    else:
    #        qubit = f

    qubit = fqbit
    #print(fqbit)
    tries = 0
    # attempt to make connections between qubits in numbered order
    #print(lqubit)
    #print(qubit < lqubit)
    while qubit < lqubit and tries < 33:
        #print(qubit)
        if qubit in wqubits:
            tries += 1
            qubits.append(qubit)
            # if "left" qubit, add 4 for straight coupling
            # otherwise, subtract 3 for diagaonal coupling
            if ((qubit % 8) < 4):
                coupling = [qubit, qubit+4]
            else:
                coupling = [qubit-3, qubit]
            # try making a connection until one is found or last qubit reached
            while (coupling[1] <= lqubit):
                if coupling in wcouplers:
                    couplings.append(coupling)
                    if ((qubit % 8) < 4):
                        qubit = ((qubit+4)%8)+fqbit
                    else:
                        qubit = ((qubit+5)%8)+fqbit
                    break
                else:
                    if ((qubit % 8) < 4):
                        coupling = list(map(add, coupling, [0, 1]))
                    else:
                        coupling = list(map(add, coupling, [1, 0]))

                    if coupling[1] == lqubit-1:
                        if ((qubit % 8) < 4):
                            qubit = ((qubit+4)%8)+fqbit
                        else:
                            qubit = ((qubit+5)%8)+fqbit
        else:
            tries += 1
            if ((qubit % 8) < 4):
                qubit = ((qubit+4)%8)+fqbit
            else:
                qubit = ((qubit+5)%8)+fqbit

    for qubit in qubits:
        H[(qubit, qubit)] = hweight
    for coupling in couplings:
        H[tuple(coupling)] = Jweight

    return H

def ml_measurement(probs, num_qubits, qubits=None):
    """
    Finds the most likely measurement outcome predicted from discrete probability
    distribution of n-qubits. (Relies on probs being canonicaly tensor product ordering).

    Inputs
    ---------------------------------------------------------------------
    probs: list of probability amplitudes for each composite qubit state

    Output
    ---------------------------------------------------------------------
    ml_state: reconstructed most likely state as a string
    """
    state = []

    # gets the most likely state
    max_idx = np.argmax(probs)

    # finds the correct state of each qubit with a tensor product ordering assumption
    for n in range(num_qubits):
        power = num_qubits - n
        mod = 2**power
        cut_off = mod / 2
        if (max_idx) % mod >= cut_off:
            state.append(1)
        else:
            state.append(0)

    # returns only the qubits of interest if a subset of qubits is specified
    sub_system_state = []
    if qubits:
        for (idx, qs) in enumerate(state):
            if idx in qubits:
                sub_system_state.append(qs)
        state = sub_system_state

    return state


def get_state_plot(data, figsize=(12, 8), filename=None, title='Distribution of Final States'):
    ncount = len(data)

    plt.figure(figsize=figsize)
    ax = sns.countplot(x='states', data=data)
    plt.title(title)
    plt.xlabel('State')

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                    ha='center', va='bottom')  # set the alignment of the text

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    # ax2.grid(None)

    if filename:
        plt.savefig(filename, dpi=300)

    return plt

def KL_div(diag, others):
    """
    Compares the Kullback Liebler divergence of different probability density
    functions w.r.t. direct diagonlization.

    Inputs
    -------------------------------------------------------------------------------
    diag: list of probs of each state obtained via direct diagonlization of final H
    others: dict of the form {'name': probs}

    Outputs
    -------------------------------------------------------------------------------
    KL_div = {'name': KL value}
    """

    return {name: entropy(diag, others[name]).flatten() for name in others}

def random_partition(dictrep_H):
    """
    Creates a random partition of the H from 1 to n-1 qubits.

    Input
    ---------------------------------------------------------
    dictrep_H: a Hamiltonian represented in the dictrep class

    Output
    ---------------------------------------------------------
    Returns a dictionary with the following elements:
    *dictHR: random partition of H (paritions qubits and THEN
    looks at random set of couplers that involve these qubits)

    *Rqubits: qubits that 'belog' to HR partition

    *dictHF: the complement of dictHR w.r.t. H

    Note: not very efficient implementation, but I don't want to
    perform random choice of qubits and couplers at the same time,
    as this leads to issues of getting random HR couplings between
    only HF qubits...
    """
    H = dictrep_H.H
    Hgraph = dictrep_H.graph
    qubit_list = dictrep_H.qubits
    nqubits = dictrep_H.nqubits

    # find HR, a random partition of H
    rand_int = random.randint(1, nqubits-1)
    rand_qubits = random.sample(qubit_list, rand_int)
    dictHR = {}
    for qubit in rand_qubits:
        dictHR.update({(qubit, qubit): H[(qubit, qubit)]})
        neighbors = list(Hgraph.neighbors(qubit))
        rand_int = random.randint(1, len(neighbors))
        rand_neighbors = random.sample(neighbors, rand_int)
        for rn in rand_neighbors:
            rand_coupler = tuple(sorted((qubit, rn)))
            #dictHR.update({rand_coupler: H[rand_coupler]})
            dictHR[rand_coupler] = H[rand_coupler]

    # create the complementary dictionary of dictHR
    dictHF = {}
    for key, value in H.items():
        if key not in dictHR:
            dictHF[key] = value
            if key[0] == key[1]:
                dictHR[key] = 0
        else:
            if key[0] == key[1]:
                dictHF[key] = 0

    return {'HR': dictHR, 'Rqubits': rand_qubits, 'HF': dictHF}

def gs_calculator(H, etol=1e-8, stol=1e-12):
    """
    Computes the (possibly degenerate) ground state of an input
    Hamiltonian H.

    H: a QuTIP defined Hamitltonian
    gs: ground-state in QuTIP style
    """
    energies, states = H.eigenstates()
    degeneracy = 1
    lowest_E = energies[0]
    for n in range(1,len(energies)):
        if abs(energies[n]-lowest_E) < etol:
            degeneracy += 1

    gs = states[0]
    for n in range(1, degeneracy):
        gs = gs + states[n]
    gs = gs.tidyup(stol)
    gs = gs.unit()

    return gs

def dense_connect_2000Q(chipdata, qi, R, C, hval, Jval):
    '''
    Takes in the "chipdata" as a dictionary that contains working qubits and couplers and
    finds (if possible) the longest unbroken chain connecting q1 to q2.

    chipdata: {'wqubits': [q1, q2, ...], 'couplers' [[q1, q2], [q2, q3]...]}
    qi: integer specifiying lowest # qubit (initial) being connected
    R: integer specifying how many rows to chain including q1 row
    C: interger specifying how many columns to chain including q1 column
    '''
    dictH = {}
    # pre-processing steps
    # --------------------------
    # for the 2000Q, the following parameters can be used
    ucsize, rows, columns = 8, 16, 16
    wqubits = chipdata['wqubits']
    wcouplers = chipdata['wcouplers']
    G = nx.Graph()
    G.add_edges_from(wcouplers)
    # first, check that qi is actually in the graph
    if qi not in wqubits:
        return("Error: qi is not in the list of working qubits for this graph.")
    # compute initial and final unit cells
    ui = floor(qi/ucsize)
    uf = (ui+R*C)-1 + (columns-C)*(R-1)
    # print out the unit cells trying to be connected for users
    print("Going to try and connect unit cell {} with unit cell {}".format(ui, uf))
    # now, ensure ui and uf can, in fact, be connected
    connections = 0
    for q in range(uf*8, uf*8 + 7):
        try:
            nx.shortest_path(G, qi, q)
            connections += 1
            break
        except:
            continue
    if connections == 0:
        return("There are no paths that connect unit cell {} with unit cell {}.".format(ui, uf))

    # densely connect within unit cells and "heuristically" connect adjacent cells
    # aka try sensical connections and ensure connected afterwards
    for row in range(0, R):
        # find the first and last unit cell in this column
        uci = ui + row*columns
        ucf = uci + (C-1)
        # iterate to last unit cell
        for uc in range(uci, ucf+1):
            # find working qubits in uc
            ucq = [q + uc*8 for q in range(8)]
            working_ucq = [q for q in ucq if q in wqubits]
            qi = min(working_ucq)
            qf = max(working_ucq)

            # create a unit cell subgraph and find longest chain by exhaustive enumeration
            ucG = nx.Graph()
            ucG.add_edges_from(G.subgraph(working_ucq).edges())
            paths = nx.all_simple_paths(ucG, qi, qf)
            node_path = max(paths, key=len)
            dictH.update({(q, q): hval for q in node_path})
            best_path = []
            for i in range(len(node_path)-1):
                best_path.append((node_path[i], node_path[i+1]))
            dictH.update({key: Jval for key in best_path})

            # connect uc to neighbor to the right
            used_rqubits = []
            if uc != ucf:
                # get the qubits that connect to the right
                rqubits = [q for q in working_ucq if q % 8 > 3]
                for q in rqubits:
                    if [q, q+8] in wcouplers:
                        dictH.update({(q, q+8): Jval, (q, q): hval})
                        used_rqubits.append(q)

                if not any(q in node_path for q in rqubits):
                    return("Heuristic chaining failed. There are no inter-unit cell connections" +
                    " between uc {} and {}".format(uc, uc+1))

            # connect uc to neighbor down below
            used_dqubits = []
            if row != (R-1):
                # get the qubits that connect down
                dqubits = [q for q in working_ucq if q % 8 <= 3]
                for q in dqubits:
                    if [q, q+128] in wcouplers:
                        dictH.update({(q, q+128): Jval, (q, q): hval})
                        used_dqubits.append(q)

                if not any(q in node_path for q in dqubits):
                    return("Heuristic chaining failed. There are no inter-unit cell connections" +
                    " between uc {} and {}".format(uc, uc+16))

    qubits_used = len([(key, value) for key, value in dictH if key == value])
    print("Successfully created a dense chain with {} qubits.".format(qubits_used))

    return dictH

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
    direction = kwargs.get('direction', '')
    sa = kwargs.get('sa', '')
    tp = kwargs.get('tp', '')
    tq = kwargs.get('tq', '')

    # turn discretization into samplerate multiplier
    samplerate = 1 / discretization

    if direction == 'forward' or direction == '':

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

    elif direction == 'reverse':
        # if no pause, do standard 'reverse' anneal
        if not tp:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sa - 1) / ta
            ba = 1
            mq = (1 - sa) / tq
            bq = (sa*(ta + tq) - ta)/tq

            # create a list of times where sampling for anneal/ quench proportional to time there
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tq, int((ta+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tq)],
                                [lambda t: ba + ma*t, lambda t: bq + mq*t])

        # otherwise, include pause
        else:
            # determine slopes and y-intercept (bq) to create piece-wise function
            ma = (sa - 1) / ta
            ba = 1
            mp = 0
            mq = (1 - sa) / tq
            bq = (sa*(ta + tp + tq) - (ta + tp))/tq

            # create a list of times with samplerate elements from 0 and T = ta + tp + tq
            t = reduce(np.union1d, (np.linspace(0, ta, int((ta+1)*samplerate)),
                                    np.linspace(ta+.00001, ta+tp, int((ta+tp+1)*samplerate)),
                                    np.linspace(ta+tp+.00001, ta+tp+tq, int((ta+tp+tq+1)*samplerate))))

            # create a piece-wise-linear (PWL) s(t) function defined over t values
            sfunc = np.piecewise(t, [t <= ta, (ta < t) & (t <= ta+tp),(ta+tp < t) & (t <= ta+tp+tq)],
                                 [lambda t:ba + ma*t, lambda t: sa, lambda t: bq + mq*t])



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
