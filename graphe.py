import copy
import random


class Graphe:
    def __init__(self, sommet=None, arrete=None, fic=None):
        self._it = 0
        if fic != None:
            self._V = []
            self._E = {}
            with open(fic, "r") as f:
                lignes = [l.strip() for l in f.readlines() if l.strip()]

                i_sommets = lignes.index("Sommets")
                i_aretes = lignes.index("Aretes")

                for l in lignes[i_sommets + 1: i_aretes - 2]:
                    self._V.append(l)
                    self._E[l] = []

                aretes = lignes[i_aretes + 1:]
                for l in aretes:
                    x, y = l.split()
                    if x not in self._E:
                        self._E[x] = []
                    self._E[x].append(y)
                    if y not in self._E:
                        self._E[y] = []
                    self._E[y].append(x)
                return
        if sommet is None:
            sommet = []
        if arrete is None:
            arrete = {}
        self._V = sommet
        self._E = arrete

    def supprimeSommet(self, v):
        G = Graphe(self._V.copy(), copy.deepcopy(self._E))
        if v not in G._V:
            return G
        for s in G._E.get(v, []):
            G._E[s].remove(v)
        G._E.pop(v, None)
        G._V.remove(v)
        return G

    def supprimerEnsemble(self, E):
        G = Graphe(self._V.copy(), copy.deepcopy(self._E))
        for v in E:
            if v not in G._V:
                continue
            for s in G._E.get(v, []):
                G._E[s].remove(v)
            G._E.pop(v, None)
            G._V.remove(v)
        return G

    def degresTousSommets(self):
        return [len(self._E.get(s, [])) for s in self._V]

    def degresMax(self):
        max = None
        degMax = -1
        for v in self._V:
            if len(self._E.get(v, [])) > degMax:
                max = v
                degMax = len(self._E.get(v, []))
        return max

    def ajouter(self, x, y):
        if x not in self._V:
            self._V.append(x)
        if x not in self._E:
            self._E[x] = []
        if y not in self._V:
            self._V.append(y)
        if y not in self._E:
            self._E[y] = []
        self._E[x].append(y)
        self._E[y].append(x)

    def __str__(self):
        res = str(self._V) + "\n"
        for s, v in self._E.items():
            res += str(s) + " -> " + str(v) + "\n"
        return res

    def __len__(self):
        return len(self._V)

    def __iter__(self):
        self._it = 0
        return self

    def __next__(self):
        if self._it >= len(self._V):
            raise StopIteration
        s = self._V[self._it]
        self._it += 1
        return s

    @staticmethod
    def generation(n, p):
        G = Graphe()
        G._V = [i for i in range(n)]
        for i in range(n):
            G._E[i] = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() <= p:
                    G._E[i].append(j)
                    G._E[j].append(i)
        return G


def algo_couplage(G):
    C = set()
    for (i, j) in G._E.items():
        if i in C:
            continue
        for e in j:
            if e not in C:
                C.add(i)
                C.add(e)
                break
    return C


def reste_arrete(G):
    for v in G._E.values():
        if len(v) > 0:
            return True
    return False


def algo_glouton(G):
    C = []
    G2 = G
    while reste_arrete(G2):
        v = G2.degresMax()
        if len(G2._E.get(v, [])) == 0:
            break
        C.append(v)
        G2 = G2.supprimeSommet(v)
    return C


G = Graphe([], {}, "exempleinstance.txt")
print(G)
print(algo_couplage(G))
print(algo_glouton(G))
