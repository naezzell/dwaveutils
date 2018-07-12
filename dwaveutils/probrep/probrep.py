from abc import ABCMeta, abstractmethod


class ProbRep(object, metaclass=ABCMeta):

    """
    An abstract class ProRep (ie ProblemRepresentation) that enforces
    properties and methods of DWave problem representation classess
    """

    @abstractmethod
    def __init__(self, qpu, vartype, encoding):

        # this should be exactly as dwave config
        self.qpu = qpu

        # varytype must be qubo or ising
        vartype = vartype.lower()
        assert (vartype == 'qubo' or vartype == 'ising'), ("vartype must be qubo or ising")
        self.vartype = vartype.lower()

        # encoding must be logical or direct
        encoding = encoding.lower()
        assert(encoding == 'logical' or encoding == 'direct'), ("encoding must be logical or direct")
        self.encoding = encoding.lower()

    @abstractmethod
    def call_annealer(self):
        pass

    @abstractmethod
    def visualize_graph(self):
        pass

    @abstractmethod
    def get_QUBO_rep(self):
        pass

    @abstractmethod
    def get_Ising_rep(self):
        pass
