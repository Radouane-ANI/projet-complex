import time
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import math


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


def comparer_algos_naifs(Nmax, p, num_instances=10):
    """
    Compare le temps de calcul et la qualité des solutions pour deux algorithmes.

    :param Nmax: Taille maximale du graphe à tester.
    :param p: Probabilité de création d'une arête dans les graphes aléatoires.
    :param num_instances: Nombre d'instances à générer pour chaque taille de graphe.
    """
    tailles = np.linspace(Nmax / 10, Nmax, 10, dtype=int)

    temps_moyens = {'branchbound': [], 'branchbound amélioré': []}
    qualite_moyenne = {'branchbound': [], 'branchbound amélioré': []}

    print(
        f"Début de la comparaison pour Nmax={Nmax}, p={p}, num_instances={num_instances}")

    for n in tailles:
        print(f"  Test pour n = {n}...")
        temps_instance = {'branchbound': [], 'branchbound amélioré': []}
        qualite_instance = {'branchbound': [], 'branchbound amélioré': []}

        for _ in range(num_instances):
            G = Graphe.generation(n, p)

            # Test de algo_couplage
            start_time = time.time()
            solution_couplage = algo_branchement_bornes(G)
            end_time = time.time()
            temps_instance['branchbound'].append(end_time - start_time)
            qualite_instance['branchbound'].append(len(solution_couplage))

            # Test de algo_glouton
            start_time = time.time()
            solution_glouton = algo_branchement_bornes_upg(G)
            end_time = time.time()
            temps_instance['branchbound amélioré'].append(end_time - start_time)
            qualite_instance['branchbound amélioré'].append(len(solution_glouton))

        temps_moyens['branchbound'].append(np.mean(temps_instance['branchbound']))
        temps_moyens['branchbound amélioré'].append(np.mean(temps_instance['branchbound amélioré']))
        qualite_moyenne['branchbound'].append(
            np.mean(qualite_instance['branchbound']))
        qualite_moyenne['branchbound amélioré'].append(np.mean(qualite_instance['branchbound amélioré']))

    # --- Section de traçage des graphiques ---

    # 1. Graphique du temps de calcul (échelle linéaire)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(tailles, temps_moyens['branchbound'], 'o-', label='Algo BranchBound')
    plt.plot(tailles, temps_moyens['branchbound amélioré'], 's-', label='Algo branchbound amélioré')
    plt.xlabel("Taille du graphe (n)")
    plt.ylabel("Temps de calcul moyen (s)")
    plt.title("Temps de calcul (Échelle Linéaire)")
    plt.legend()
    plt.grid(True)

    # 2. Graphique du temps de calcul (échelle Log-Log pour complexité polynomiale)
    plt.subplot(2, 2, 2)
    plt.loglog(tailles, temps_moyens['branchbound'], 'o-', label='Algo Couplage')
    plt.loglog(tailles, temps_moyens['branchbound amélioré'], 's-', label='Algo Glouton')
    plt.xlabel("log(Taille du graphe)")
    plt.ylabel("log(Temps de calcul moyen)")
    plt.title("Temps de calcul (Échelle Log-Log)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 3. Graphique de la qualité de la solution
    plt.subplot(2, 2, 3)
    plt.plot(tailles, qualite_moyenne['branchbound'], 'o-', label='Algo Couplage')
    plt.plot(tailles, qualite_moyenne['branchbound amélioré'], 's-', label='Algo Glouton')
    plt.xlabel("Taille du graphe (n)")
    plt.ylabel("Taille moyenne de la couverture")
    plt.title("Qualité des solutions")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def algo_branchement(G):
    C = set()
    for (i, j) in G._E.items():
        for e in j:
            res1 = algo_branchement(G.supprimeSommet(i))
            res1.add(i)
            res2 = algo_branchement(G.supprimeSommet(e))
            res2.add(e)
            if len(res1) < len(res2):
                C = res1
            else:
                C = res2
            return C
    return C



def calculer_couplage_max(G):
    """Retourne un couplage maximal (ensemble d'arêtes disjointes)"""
    couplage = []
    sommets_utilises = set()
    
    for i in G._E:
        if i in sommets_utilises:
            continue
        for j in G._E[i]: # tant que les sommets ne sont pas utilisés, ajouter aretes
            if j not in sommets_utilises:
                couplage.append((i, j))
                sommets_utilises.add(i)
                sommets_utilises.add(j)
                break
    return couplage

def calculer_borne_inf(G):
    """Calcule la borne inférieure max{b1, b2, b3}"""
    n = len(G._V)
    
    # compter le nombre d'aretes m
    m = sum(len(voisins) for voisins in G._E.values()) // 2
    
    if m == 0:
        return 0
    
    # b1 = ⌈m/delta⌉
    delta = max(len(G._E.get(v, [])) for v in G._V) if G._V else 1
    b1 = math.ceil(m / delta) if delta > 0 else 0
    
    # b2 = |M| (taille du couplage maximal)
    couplage = calculer_couplage_max(G)
    b2 = len(couplage)
    
    # b3 = (2n-1-sqrt((2n-1)²-8m))/2
    discriminant = (2*n - 1)**2 - 8*m
    if discriminant >= 0:
        b3 = (2*n - 1 - math.sqrt(discriminant)) / 2
        b3 = math.ceil(b3)
    else:
        b3 = 0
    
    return max(b1, b2, b3)

def algo_branchement_bornes(G, meilleure_sol=None):
    """
    Algorithme de branchement avec élagage par bornes
    """
    # si plus d'arête, retourner ensemble vide
    if not reste_arrete(G):
        return set()
    
    # calculer borne inférieure
    borne_inf = calculer_borne_inf(G)
    
    # élagage si la borne inf >= meilleure solution connue, on abandonne
    if meilleure_sol is not None and borne_inf >= len(meilleure_sol):
        return meilleure_sol
    
    # calculer une solution réalisable avec algo_couplage
    sol_realisable = algo_couplage(G)
    
    # mettre à jour la meilleure solution
    if meilleure_sol is None or len(sol_realisable) < len(meilleure_sol):
        meilleure_sol = sol_realisable.copy()
    
    # choisir une arête à brancher (prendre le sommet de degré max)
    v = G.degresMax()
    if v is None or len(G._E.get(v, [])) == 0:
        return meilleure_sol
    
    # prendre un voisin pour brancher
    voisin = G._E[v][0]
    
    # branche 1 on prend v dans la couverture
    G1 = G.supprimeSommet(v)
    res1 = algo_branchement_bornes(G1, meilleure_sol)
    res1.add(v)
    
    # màj meilleure solution
    if len(res1) < len(meilleure_sol):
        meilleure_sol = res1.copy()
    
    # branche 2: on prend le voisin dans la couverture
    G2 = G.supprimeSommet(voisin)
    res2 = algo_branchement_bornes(G2, meilleure_sol)
    res2.add(voisin)
    
    # retourner la meilleure des deux branches
    if len(res1) < len(res2):
        return res1
    else:
        return res2


def algo_branchement_bornes_upg(G, meilleure_sol=None):
    """
    Algorithme de branchement avec élagage par bornes
    """
    # si plus d'arête, retourner ensemble vide
    if not reste_arrete(G):
        return set()
    
    # calculer borne inférieure
    borne_inf = calculer_borne_inf(G)
    
    # élagage si la borne inf >= meilleure solution connue, on abandonne
    if meilleure_sol is not None and borne_inf >= len(meilleure_sol):
        return meilleure_sol
    
    # calculer une solution réalisable avec algo_couplage
    sol_realisable = algo_couplage(G)
    
    # mettre à jour la meilleure solution
    if meilleure_sol is None or len(sol_realisable) < len(meilleure_sol):
        meilleure_sol = sol_realisable.copy()
    
    # choisir une arête à brancher (prendre le sommet de degré max)
    v = G.degresMax()
    if v is None or len(G._E.get(v, [])) == 0:
        return meilleure_sol
    
    # prendre un voisin pour brancher
    voisin = G._E[v][0]
    
    # branche 1 on prend v dans la couverture
    G1 = G.supprimeSommet(v)
    res1 = algo_branchement_bornes(G1, meilleure_sol)
    res1.add(v)
    
    # màj meilleure solution
    if len(res1) < len(meilleure_sol):
        meilleure_sol = res1.copy()
    
    # branche 2: on prend le voisin dans la couverture
    G2 = G.supprimeSommet(voisin)
    for voisin_v in G._E[v]: # amélioration, on prend voisins de v et on les supprime
        G2 = G2.supprimeSommet(voisin_v) 
    res2 = algo_branchement_bornes(G2, meilleure_sol) #on résout le nouveau
    res2.add(voisin)
    for voisin_v in G._E[v]:
        res2.add(voisin_v) 
    
    # retourner la meilleure des deux branches
    if len(res1) < len(res2):
        return res1
    else:
        return res2

def couverture_valide(G, C):
    """Vérifie que C est bien une couverture de G"""
    for sommet in G._E:
        for voisin in G._E[sommet]:
            if sommet not in C and voisin not in C:
                return False
    return True


if __name__ == '__main__':
    N_MAX = 30
    PROBABILITE_ARETE = 0.1
    NOMBRE_INSTANCES = 10


    """  Test avec fichier couplage et glouton :
    G = Graphe(fic="exempleinstance.txt")
    print("Graphe chargé:", G)
    print("Couplage:", algo_couplage(G))
    print("Glouton:", algo_glouton(G)) """


    """G = Graphe.generation(10, 0.3)
    print("Graphe",G)
    print("Couplage:", algo_couplage(G))
    print("Glouton:", algo_glouton(G)) """
    

    
    comparer_algos_naifs(N_MAX, PROBABILITE_ARETE, NOMBRE_INSTANCES)
    G = Graphe(fic="exempleinstance.txt")
    print(algo_branchement(G))
 

    """
    G = Graphe(fic="exempleinstance.txt")
    print("Graphe",G)
    for s, d in zip(G._V, G.degresTousSommets()):
        print(f"Sommet {s} : degré {d}")
    sol = algo_glouton(G)
    print("Glouton",sol)
    sol_c = algo_couplage(G)
    print("Couplage",sol_c)
    sol_branch = algo_branchement(G)
    print("Branchement",sol_branch)
    print("Solution valide glouton ?", couverture_valide(G, sol))
    print("Solution valide couplage ?", couverture_valide(G, sol_c))
    print("Solution valide branchement ?", couverture_valide(G, sol_branch)) """

    G = G = Graphe.generation(13, 0.5)
    print("Graphe :", G)
    print(f"Borne inférieure calculée (b1, b2, b3, max) : {calculer_borne_inf(G)}")
    sol_bornes = algo_branchement_bornes(G)
    sol_branch = algo_branchement_bornes_upg(G)
    print("solution branchement avec bornes",sol_branch, "taille =",len(sol_branch))
    print("Solution branchement avec bornes amélioré :", sol_bornes, "taille =", len(sol_bornes)) 
    # j'ai pas essayé avec les plots de comparer_algos_naifs()





    
