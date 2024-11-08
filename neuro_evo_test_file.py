"""This was built to allow for a place to experiment with the previous modules to see if I could
achieve my desired results. For the most part this was successful with achieving convergence on a
decent solution. Future plans could include cleaning the modules so they run faster as well as
finding working algorithms for non topological neuro evolution to implement crossover and mutation
better"""

import matplotlib.pyplot as plt
import population_environment as PE

goal = [10,10]
env = PE.Env(20, 20, 30)
population = PE.Population(1000, env, goal)
env.set_population(population)
CHECK = True
skeletonNN = PE.gen_neural_net(2,4,4,3)
FRAME = 0
GENERATION = 0
best_fitness = []
average_fitnesses = []
open("most_recent_run_gens.txt", "w").close



while CHECK:
    env.step(skeletonNN)
    best_pos = population.pop[0].position
    population.selection_recombo(300, mc = .2)#mc is mutation chance

    GENERATION +=1
    best_fitness.append(sorted(population.pop)[0].fitness)
    average_fitnesses.append(population.average_fitness)
    env.reset()

    print(GENERATION)
    with open("Most_recent_run_Gens.txt", "a") as file:
        file.write(f"GENERATION: {GENERATION}\n")
        file.write(f"It best fitness: {best_fitness[-1]}\n")
        file.write(f"It fitness Average: {population.average_fitness}\n")
        file.write(f"Best Position: {best_pos}\n")
        file.write("\n")
        file.close()

    if GENERATION == 100:
        plt.plot(range(int(GENERATION)),average_fitnesses)
        plt.ylim(0,1)
        plt.xlabel("Generations")
        plt.ylabel("it Fitness")
        plt.show()
        break
