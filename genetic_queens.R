# N-Queens Solution with Genetic Algorithms in R
# Victhor S. Sartório

library(magrittr)


gen_pop <- function(pop_size, chrom_size) {
    # Returns a matrix with shape (pop_size, chrom_size) filled with
    # values sampled from a discrete uniform distribution with
    # support in {1, 2, ..., chrom_size}.

    sample.int(chrom_size, chrom_size*pop_size, TRUE) %>%
    matrix(nrow = pop_size, ncol = chrom_size)
}


.fitness <- function(chrom) {
    # Returns a measure of fitness for a particular chromossome in
    # the inverval [0, 1] where 1 means it's a perfect solution.

    chrom_size = length(chrom)
    attacks = 0

    for (i in 1:(chrom_size-1)) {
        queen = chrom[i]

        for (j in (i+1):chrom_size) {
            attacker = chrom[j]

            if (queen == attacker) {
                attacks = attacks + 1
                next
            }
            if (queen + i - j == attacker || queen - i + j == attacker) {
                attacks = attacks + 1
                next
            }
        }
    }

    1 - attacks * 2 / chrom_size / (chrom_size-1)
}

fitness <- function(pop) {
    # Applies .fitness to every chromossome in the population pop.

    apply(pop, 1, .fitness)
}


crossover <- function(pop, mut_rate, fitvec = NULL) {
    pop_size   = nrow(pop)
    chrom_size = ncol(pop)

    # If the fitness was not already calculated, calculate it
    if (is.null(fitvec)) {
        fitvec = fitness(pop)
    }

    # Preallocate the new population
    newpop = matrix(0, nrow = pop_size, ncol = chrom_size)

    # Extract random numbers we deterministically know will be needed
    # to save on function call overhead and possibly take advantage
    # of vectorization in the function implementations.
    sampler = sample.int(pop_size, 2*pop_size, TRUE, fitvec)
    crosspt = sample.int(chrom_size - 2, pop_size, TRUE) + 1
    mutpoll = runif(pop_size)

    for (i in 1:pop_size) {
        # Select parents and generate children
        parent1 = pop[sampler[2*i-1], ]
        parent2 = pop[sampler[2*i], ]
        newpop[i, ] = c(parent1[1:crosspt[i]], parent2[(crosspt[i]+1):chrom_size])

        # Mutation step
        if (mut_rate > mutpoll[i]) {
            aux = sample.int(chrom_size, 2, TRUE)
            newpop[i,aux[1]] = aux[2]
        }
    }

    newpop
}

solve_nq <- function(pop_size, chrom_size, mut_rate, stopcond = 0.999) {
    if (pop_size <= 10 || chrom_size <= 3 || mut_rate < 0 || mut_rate > 1) {
        stop("Invalid arguments to 'solve_nq'")
    }

    pop = gen_pop(pop_size, chrom_size)
    fit = fitness(pop)

    while (!any(fit > stopcond)) {
        pop = crossover(pop, mut_rate, fit)
        fit = fitness(pop)
    }

    pop[which(fit > stopcond)[1], ]
}
