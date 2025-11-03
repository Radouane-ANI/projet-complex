import time
import matplotlib.pyplot as plt
import numpy as np
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


def comparer_algos(algo1, nom_algo1, algo2, nom_algo2, Nmax, p, num_instances=10):
    """
    Compare le temps de calcul et la qualité des solutions pour deux algorithmes.

    :param algo1: La première fonction d'algorithme à tester.
    :param nom_algo1: Le nom de la première fonction pour les légendes.
    :param algo2: La deuxième fonction d'algorithme à tester.
    :param nom_algo2: Le nom de la deuxième fonction pour les légendes.
    :param Nmax: Taille maximale du graphe à tester.
    :param p: Probabilité de création d'une arête dans les graphes aléatoires.
    :param num_instances: Nombre d'instances à générer pour chaque taille de graphe.
    """
    tailles = np.linspace(Nmax / 10, Nmax, 10, dtype=int)

    temps_moyens = {nom_algo1: [], nom_algo2: []}
    qualite_moyenne = {nom_algo1: [], nom_algo2: []}

    print(
        f"Début de la comparaison de '{nom_algo1}' vs '{nom_algo2}' pour Nmax={Nmax}, p={p}, num_instances={num_instances}")

    for n in tailles:
        print(f"  Test pour n = {n}...")
        temps_instance = {nom_algo1: [], nom_algo2: []}
        qualite_instance = {nom_algo1: [], nom_algo2: []}

        for _ in range(num_instances):
            G = Graphe.generation(n, p)

            # Test de algo1
            start_time = time.time()
            solution_algo1 = algo1(G)
            end_time = time.time()
            temps_instance[nom_algo1].append(end_time - start_time)
            qualite_instance[nom_algo1].append(len(solution_algo1))

            # Test de algo2
            start_time = time.time()
            solution_algo2 = algo2(G)
            end_time = time.time()
            temps_instance[nom_algo2].append(end_time - start_time)
            qualite_instance[nom_algo2].append(len(solution_algo2))

        temps_moyens[nom_algo1].append(np.mean(temps_instance[nom_algo1]))
        temps_moyens[nom_algo2].append(np.mean(temps_instance[nom_algo2]))
        qualite_moyenne[nom_algo1].append(
            np.mean(qualite_instance[nom_algo1]))
        qualite_moyenne[nom_algo2].append(np.mean(qualite_instance[nom_algo2]))

    # --- Section de traçage des graphiques ---

    label1 = f'Algo {nom_algo1.capitalize()}'
    label2 = f'Algo {nom_algo2.capitalize()}'

    # Augmentation de la taille pour un meilleur affichage des 4 plots
    plt.figure(figsize=(14, 10))

    # 1. Graphique du temps de calcul (Échelle linéaire) pour Algo 1
    plt.subplot(2, 2, 1)
    plt.plot(tailles, temps_moyens[nom_algo1],
             'o-', label=label1, color='blue')
    plt.xlabel("Taille du graphe (n)")
    plt.ylabel("Temps de calcul moyen (s)")
    plt.title(f"Temps de calcul - {nom_algo1.capitalize()} (Linéaire)")
    plt.grid(True)

    # 2. Graphique du temps de calcul (Échelle linéaire) pour Algo 2
    plt.subplot(2, 2, 2)
    plt.plot(tailles, temps_moyens[nom_algo2], 's-', label=label2, color='red')
    plt.xlabel("Taille du graphe (n)")
    plt.ylabel("Temps de calcul moyen (s)")
    plt.title(f"Temps de calcul - {nom_algo2.capitalize()} (Linéaire)")
    plt.grid(True)

    # 3. Graphique du temps de calcul (échelle Log-Log pour complexité polynomiale) - Décalé à la position 3
    plt.subplot(2, 2, 3)
    plt.loglog(tailles, temps_moyens[nom_algo1], 'o-', label=label1)
    plt.loglog(tailles, temps_moyens[nom_algo2], 's-', label=label2)
    plt.xlabel("log(Taille du graphe)")
    plt.ylabel("log(Temps de calcul moyen)")
    plt.title("Temps de calcul (Échelle Log-Log - Comparaison)")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # 4. Graphique de la qualité de la solution - Décalé à la position 4
    plt.subplot(2, 2, 4)
    plt.plot(tailles, qualite_moyenne[nom_algo1], 'o-', label=label1)
    plt.plot(tailles, qualite_moyenne[nom_algo2], 's-', label=label2)
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


if __name__ == '__main__':
    N_MAX = 500
    PROBABILITE_ARETE = 0.1
    NOMBRE_INSTANCES = 10

    # comparer_algos(algo_couplage, 'Couplage', algo_glouton,
    #                'Glouton', N_MAX, PROBABILITE_ARETE, NOMBRE_INSTANCES)
    G = Graphe(fic="exempleinstance.txt")
    print(algo_branchement(G))
