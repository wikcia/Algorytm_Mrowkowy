import time
from collections import defaultdict
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import random
import tsplib95

problem = tsplib95.load(
    r'C:\Users\Sandra\Desktop\META2\datasets\MiastaSYMM50\MiastaSYMM50.tsp')

class AntColonyAlgorithm:
    def __init__(self, problem_nodes: Callable[[Any, Any], Union[float, int]],
                 time=10 * 60,   # czas dzialania
                 ant_count=64,    # liczba mrowek
                 ant_speed=1,    # jak szybko mrowki chodza
                 distance_power=1,  # power to which distance affects pheromones
                 pheromone_power=1.25,  # do takiej potegi podnosimy feromony
                 best_path_smell=2,  # o tyle mnozymy feromony na najlepszej sciezce
                 start_smell=10,  # liczba feromonow na start
                 ):

        self.problem_nodes = problem_nodes
        self.time = int(time)
        self.ant_count = int(ant_count)
        self.ant_speed = int(ant_speed)
        self.distance_power = float(distance_power)
        self.pheromone_power = float(pheromone_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell = float(start_smell)

        self._initalized = False

    def solve_initialize(self, problem_path: List[Any], ) -> None:
        # Tablica zawierajaca informacje ile z jakiegos miasta do kazdego innego miasta jest drogi
        self.distances = defaultdict(dict)
        for source in problem_path:
            for dest in problem_path:
                self.distances[source][dest] = self.problem_nodes(source, dest)

        # 1 / (1 + dlugosc drogi)
        # pozniej bedziemy to mnozyc razy liczbe feromonow
        # przykladowo distance_cost = 1 / 8
        # ilosc feromonow = 10
        # to ilosc feromonow na jednostke drogi to 10/8
        # im krotsza sciezka tym wiecej feromonow mrowka zostawi na drodze
        self.distance_cost = defaultdict(dict)
        for source in problem_path:
            for dest in problem_path:
                self.distance_cost[source][dest] = 1 / (1 + self.distances[source][dest])

        # Tablica feromonów czyli ile jest feromonow z jednej sciezki do kazdej innej
        self.pheromones = defaultdict(dict)
        for source in problem_path:
            for dest in problem_path:
                self.pheromones[source][dest] = self.start_smell

        self._initalized = True

    def solve(self, problem_path: List[Any]) -> List[Tuple[int, int]]:
        if not self._initalized:
            self.solve_initialize(problem_path)

        ants = {
            "distance": np.zeros((self.ant_count,)).astype('int32'), # dystans mrowek miedzy miastem, w ktorym dopiero co byly a miastem ktory dopiero odwiedzily
            "path": [[problem_path[0]] for n in range(self.ant_count)],  # miasta ktore mrowki odwiedzily
            "remaining": [set(problem_path[1:]) for n in range(self.ant_count)], # miasta ktore mrowki musza jeszcze odwiedzic
            "path_cost": np.zeros((self.ant_count,)).astype('int32'),  # koszt calej drogi
        }

        best_path = None
        best_path_cost = np.inf
        time_start = time.perf_counter()
        while True:
            # mrowki podrozujace to te ktory ich dystans > 0 (gdzie dystans to dystans miedzy poprzednim miastem a obecnym)
            ants_travelling = (ants['distance'] > 0)
            ants['distance'][ants_travelling] -= self.ant_speed # mrowka podrozuje, jesli dystans <= 0 to mrowka dotarla
            # do jakiegos miasta lub mrowiska (obliczamy pozniej prawdopodobienstwo do jakiego dotarla)
            if all(ants_travelling):
                continue # jesli wszystkie podrozuja to wracamy do poczatku petli while

            # Sprawdzamy ktore mrowki wrocily
            ants_arriving = np.invert(ants_travelling) # tablica zawierajaca prawda/fałsz, invert(ants_travelling) - jesli
            # mrowka podrozuje to ants_arriving zawiera false, a jeśli przestala podrozowac to zawiera true
            # czyli invert zamienia falsz na prawde i prawde na falsz
            ants_arriving_index = np.where(ants_arriving)[0]
            # przechodzimy po mrowkach ktore juz nie podrozuja
            for i in ants_arriving_index:
                # przydzielamy mrowce next node (nastepne miasto do ktorego dotarla)
                this_node = ants['path'][i][-1]
                next_node = self.next_node(ants, i)
                ants['distance'][i] = self.distances[this_node][next_node] # dystans miedzy poprzednim a nowym miastem
                ants['remaining'][i] = ants['remaining'][i] - {this_node} # odejmujemy od miast ktore zostaly do odwiedzenia obecne
                # nowe miasto
                ants['path_cost'][i] = ants['path_cost'][i] + ants['distance'][i] # doliczamy do kosztu calej drogi koszt przebyty od
                # poprzedniego miasta do obecnego nowego
                ants['path'][i].append(next_node) # dodajemy do sciezki nowe miasto

                ### mrowka wrocila do mrowiska
                if not ants['remaining'][i] and ants['path'][i][0] == ants['path'][i][-1]:
                    was_best_path = False
                    # sprawdzamy czy mrowka przeszla krotsza droge
                    if ants['path_cost'][i] < best_path_cost:
                        was_best_path = True
                        best_path_cost = ants['path_cost'][i]
                        best_path = ants['path'][i]
                        print("path_cost:", int(ants['path_cost'][i]))

                    # rozpylamy feromony na drodze ktora przeszla mrowka
                    reward = 1
                    for path_index in range(len(ants['path'][i]) - 1):
                        this_node = ants['path'][i][path_index]
                        next_node = ants['path'][i][path_index + 1]
                        self.pheromones[this_node][next_node] += reward
                        self.pheromones[next_node][this_node] += reward
                        if was_best_path:
                            # jesli byla to najlepsza sciezka to podwajamy feromony na tej sciezce
                            self.pheromones[this_node][next_node] *= self.best_path_smell
                            self.pheromones[next_node][this_node] *= self.best_path_smell

                    # resetujemy mrowke bo wrocila do mrowiska
                    ants["distance"][i] = 0
                    ants["path"][i] = [problem_path[0]]
                    ants["remaining"][i] = set(problem_path[1:])
                    ants["path_cost"][i] = 0

            # jesli przekroczylismy czas to przerwij
            clock = time.perf_counter() - time_start
            if self.time:
                if clock > self.time:
                    break
                else:
                    continue

        return best_path

    # funkcja wybierajaca miasto do ktorego ma isc mrowka
    def next_node(self, ants, index):
        this_node = ants['path'][index][-1]

        weights = []
        weights_sum = 0
        # jesli mrowka odwiedzila kazde miasto to wroc do miasta pierwszego czyli mrowiska
        if not ants['remaining'][index]: return ants['path'][index][0]
        # przechodzimy po kazdym miescie z tych ktore mrowka moze odwiedzic
        for next_node in ants['remaining'][index]:
            if next_node == this_node: continue

            reward = self.pheromones[this_node][next_node] ** self.pheromone_power * self.distance_cost[this_node][next_node]
            weights.append((reward, next_node))
            weights_sum += reward

        # Wybierz kolejne miasto
        rand = random.random() * weights_sum
        for (weight, next_node) in weights:
            if rand > weight:
                rand -= weight
            else:
                break

        return next_node


def AntColonyRunner(cities):
    ant_colony = AntColonyAlgorithm(problem_nodes=distance)
    result = ant_colony.solve(cities)

    print("Mrowki: ", "ilosc miast: ", len(cities), "dlugosc drogi: ", path_distance(result))
    return result

def distance(city1, city2) -> float:
    cost = problem.get_weight(*(city1, city2))
    return cost

def path_distance(tour) -> int:
    cost = 0
    for i in range(len(tour) - 1):
        cost += problem.get_weight(*(tour[i], tour[i + 1]))
    cost += problem.get_weight(*(tour[-1], tour[0]))
    return cost

# -------------------- 2 OPT --------------------

def get_cost(problem: tsplib95.models.StandardProblem, tour):
    cost = 0
    for i in range(len(tour) - 1):
        cost += problem.get_weight(*(tour[i], tour[i + 1]))
    cost += problem.get_weight(*(tour[-1], tour[0]))
    return cost


def random_solve(problem):
    nodes = list(problem.get_nodes())
    np.random.shuffle(nodes)
    problem.tours.append(nodes)
    return nodes


def invert(solution: list):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbours.append(solution[:i] + solution[i:j + 1][::-1] + solution[j + 1:])
    return neighbours


def two_opt(problem):
    solution = random_solve(problem)
    solution_cost = get_cost(problem, solution)
    can_find_better_solution = True
    while can_find_better_solution:
        surrounding = invert(
            solution)  # surrounding to lista wszystkich rozwiazan do jakich mozna dojsc odwracajac ciag
        new_solution = surrounding[0]  # inicjalizujemy new_solution
        new_solution_cost = get_cost(problem, new_solution)
        for possible_new_solution in surrounding:
            current_cost = get_cost(problem, possible_new_solution)
            if current_cost < new_solution_cost:  # weights decrease
                new_solution_cost = current_cost
                new_solution = possible_new_solution
        if new_solution_cost >= solution_cost:  # nie moze znalezc lepszego rozwiazania
            can_find_better_solution = False
        else:
            solution = new_solution
            solution_cost = new_solution_cost
    return {"solution": solution,
            "cost": solution_cost}


def main():
    print(list(problem.get_nodes()))

    results = AntColonyRunner(list(problem.get_nodes()))
    print("miasta: ", results)

    print('2 opt:', two_opt(problem))


if __name__ == '__main__':
    main()