# N-Queens Solution with Genetic Algorithms in Julia
# Victhor S. Sartório

import Distributions
import StatsBase

# Generates a population containing $nchroms members of size $nqueens
# with a discrete uniform distribution with support in [1, $nqueens]
# in the form of a matrix, with each row being a member.
function genpop(nchroms::Int, nqueens::Int)
	# Check parameter validity
	if nchroms <= 9 || nqueens <= 3
		throw(DomainError())
	end

	# Set up the distribution
	U = Distributions.DiscreteUniform(1, nqueens)

	# Generate and return the matrix
	return reshape(rand(U, nchroms*nqueens), nchroms, nqueens)
end

# Calculates the fitness of each member of a population
function fitness(population::Array{Int,2})
	# Get the dimensions
	(nchroms, nqueens) = size(population)
	# Allocate return vector
	fitvec = zeros(Float64, nchroms)

	# For each member calculate fitness
	for k in 1:nchroms
		# afit measures the number of attacks, which is the opposite of fitness
		local afit::UInt = 0
		for i in 1:nqueens
			pivot = population[k,i]
			for j in i+1:nqueens
				current = population[k,j]
				if pivot == current
					afit += 1
				end
				if pivot+i-j == current || pivot-i+j == current
					afit += 1
				end
			end
		end
		# We invert it, so the less "antifitness" the higher the fitness, up to Inf
		fitvec[k] = 1/Float64(afit)
	end

	return fitvec
end

# Crossover
function crossover(population::Array{Int,2}, fitvec::Array{Float64,1}, mutrate::Float64)
	# Get the dimensions
	(nchroms, nqueens) = size(population)
	# Allocate the matrix for the new population
	newpop = zeros(Int, (nchroms, nqueens))

	# Extract random numbers we deterministically know will be needed
	# to save on function call overhead and possibly take advantage of
	# vectorization in the function implementations.
	sampler = StatsBase.sample(1:nchroms, StatsBase.WeightVec(fitvec), 2*nchroms)
		
	crosspt = rand(1:nqueens, nchroms)
	mutate = rand(MersenneTwister(), nchroms)

	# Generate new population
	for k in 1:nchroms
		dad = population[sampler[2*k-1],:]
		mom = population[sampler[2*k],:]
		pivot = crosspt[k]
		kid = [dad[1:pivot]; mom[pivot+1:nqueens]]

		if mutate[k] < mutrate
			(at, val) = rand(1:nqueens, 2)
			kid[at] = val
		end

		newpop[k,:] = kid
	end

	return newpop
end

# Run
function solvenq(n::Int, popsize::Int, mutrate::Float64)
	pop = genpop(popsize, n)
	fit = fitness(pop)
	
	while !(Inf in fit)
		pop = crossover(pop, fit, mutrate)
		fit = fitness(pop)
	end

	for k in 1:popsize
		if fit[k] == Inf
			return pop[k,:]
		end
	end
end




















