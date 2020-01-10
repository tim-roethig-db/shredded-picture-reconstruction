# Tim Röthig

import blackbox
import random
from matplotlib import pyplot as plt
from deap import base
from deap import creator
from deap import tools

oracle = blackbox.BlackBox("shredded.png", orig_path='original.png')

def evaluation(ind):
    return oracle.evaluate_solution(ind),

# Structure initializer
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def indiv(size=128):
    return creator.Individual(random.sample(range(size), size))

toolbox.register("population", tools.initRepeat, list, indiv)
toolbox.register("evaluate", evaluation)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.005)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=100)

    # CXPB  is the probability with which two individuals are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.9, 1

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    Gen = []
    Fit = []

    # Begin the evolution
    while min(fits)>0 and g<500:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # Print results
        print(" %s" % min(fits))
        print(" %s" % max(fits))
        print(" %s" % mean)
        print(" %s" % std)
        best = tools.selBest(pop, 1)[0]
        print(best)

        Gen.append(g)
        Fit.append(min(fits))

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    oracle.show_solution(best_ind)

    # Plot convergence curve
    print(Gen)
    print(Fit)
    plt.plot(Gen, Fit, '-')
    plt.show()

if __name__ == "__main__":
    main()
