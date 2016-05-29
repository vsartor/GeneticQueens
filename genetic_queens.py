# -*- coding: utf-8 -*-
"""
N-Queens Solution with Genetic Algorithms in Python
Victhor S. Sartório
"""

import numpy as np
from numba import jit


def gen_pop(pop_size, chrom_size):
    """ Returns a matrix with shape (pop_size, chrom_size) filled
    with values sampled from a discrete uniform distribution with
    support in {1, 2, ..., chrom_size}. """

    return np.random.randint(1, chrom_size + 1, (pop_size, chrom_size))


@jit
def _fitness(chrom):
    """ Returns a measure of fitness for a particular chromossome in
    the interval [0, 1] where 1 means it's a perfect solution."""

    chrom_size = chrom.shape[0]
    attacks = 0

    for i in range(0, chrom_size-1):
        queen = chrom[i]

        for j in range(i+1, chrom_size):
            attacker = chrom[j]

            if attacker == queen:
                attacks += 1
                continue

            if attacker + i - j == queen or attacker - i + j == queen:
                attacks += 1
                continue

    return 1. - attacks * 2. / float(chrom_size) / (chrom_size - 1.)


@jit
def fitness(pop):
    """ Applies _fitness to every chromossome in the population."""
    pop_size = pop.shape[0]

    fitvec = np.zeros(pop_size, np.float64)

    for i in range(pop_size):
        fitvec[i] = _fitness(pop[i])

    return fitvec


@jit
def crossover(pop, mut_rate, fitvec = None):
    pop_size, chrom_size = pop.shape

    # If the fitness was not already calculated, calculate it
    if fitvec is None:
        fitvec = fitness(pop)
    fitvec = fitvec / np.sum(fitvec)

    # Preallocate the new population
    newpop = np.zeros((pop_size, chrom_size), np.int)
    
    # Extract random numbers we deterministically know will be needed
    # to save on function call overhead and possibly take advantage
    # of vectorization in the function implementations.
    sampler = np.random.choice(np.arange(0, pop_size), 2*pop_size, True, fitvec)
    crosspt = np.random.randint(1, chrom_size, pop_size)
    mutpoll = np.random.ranf(pop_size)

    for i in range(pop_size):
        # Select parents and generate children
        parent1 = pop[sampler[2*i-1]]
        parent2 = pop[sampler[2*i]]
        _crosspt = crosspt[i]
        newpop[i][:_crosspt] = parent1[:_crosspt]
        newpop[i][_crosspt:] = parent2[_crosspt:]

        # Mutation step
        if mut_rate > mutpoll[i]:
            aux = np.random.randint(0, chrom_size, 2)
            newpop[i][aux[0]] = aux[1] + 1

    return newpop


@jit
def solve_nq(pop_size, chrom_size, mut_rate, stopcond = 0.999):
    assert pop_size >= 3 and chrom_size >= 4
    assert mut_rate >= 0 and mut_rate <= 1

    pop = gen_pop(pop_size, chrom_size)
    fit = fitness(pop)

    while not np.any(fit > stopcond):
        pop = crossover(pop, mut_rate, fit)
        fit = fitness(pop)

    return pop[fit == 1][0]
