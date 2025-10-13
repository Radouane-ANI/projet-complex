import copy


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
                    self._E[x].append(y)
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
        G._V.remove(v)
        for l in G._E.values():
            if v in l:
                l.remove(v)
        G._E.pop(v)
        return G

    def supprimerEnsemble(self, E):
        G = Graphe(self._V.copy(), copy.deepcopy(self._E))
        for v in E:
            if v not in G._V:
                continue
            G._V.remove(v)
            for l in G._E.values():
                if v in l:
                    l.remove(v)
            G._E.pop(v)
        return G

    def degresTousSommets(self):
        return [len(self._E[s]) for s in self._V]

    def degresMax(self):
        max = 0
        degMax = 0
        for v, l in self._E.items():
            if len(l) > degMax:
                max = v
                degMax = len(l)
        return max

    def ajouter(self, x, y):
        if x not in self._V:
            self._V.append(x)
            self._E[x] = []
        if y not in self._V:
            self._V.append(y)
            self._E[y] = []
        self._E[x].append(y)

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
        
    def generation(self,n,p):
        G = Graphe()
        G._V = [i for i in range(n)]
        for i in range(n):
            for j in range(i,n):
                if i not in G._E:
                    G._E[i] = []
                if i==j:
                    continue
                if random.random() <= p:
                    G._E[i].append(j)
        return G



G = Graphe([], {}, "exempleinstance.txt")
print(G)

