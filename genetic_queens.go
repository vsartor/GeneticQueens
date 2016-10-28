package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat/distuv"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func printMatrix(m mat64.Matrix, cols, decs int) {
	formatString := fmt.Sprintf("%%%d.%df", cols, decs)
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf(formatString, m.At(i, j))
		}
		fmt.Printf("\n")
	}
}

func generatePopulation(popSize, numQueens int) *mat64.Dense {
	m := mat64.NewDense(popSize, numQueens, nil)
	for i := 0; i < popSize; i++ {
		for j := 0; j < numQueens; j++ {
			m.Set(i, j, float64(rand.Intn(numQueens)))
		}
	}
	return m
}

func fitness(m *mat64.Vector) float64 {
	n, _ := m.Dims()
	attacks := 0
	for i := 0; i < n-1; i++ {
		queen := int(m.At(i, 0))
		for j := i + 1; j < n; j++ {
			other := int(m.At(j, 0))

			if queen == other || queen+i-j == other || queen-i+j == other {
				attacks++
			}
		}
	}
	return 1 - float64(attacks)*2/float64(n-1)/float64(n)
}

func fitnessVector(m *mat64.Dense) []float64 {
	n, _ := m.Dims()
	v := make([]float64, n)
	for i := 0; i < n; i++ {
		v[i] = fitness(m.RowView(i))
	}
	return v
}

func crossover(pop *mat64.Dense, mutationRate float64, fitvec []float64) *mat64.Dense {
	popSize, numQueens := pop.Dims()
	newPop := mat64.NewDense(popSize, numQueens, nil)

	// If the fitnessVector was not provided, calculate it
	if fitvec == nil {
		fitvec = fitnessVector(pop)
	}

	// Distribution from which the index of a parent is taken
	sampler := distuv.NewCategorical(fitvec, nil)

	for i := 0; i < popSize; i++ {
		// Get parents and produce child
		father := int(sampler.Rand())
		mother := int(sampler.Rand())
		crossoverPoint := rand.Intn(numQueens)
		for k := 0; k < crossoverPoint; k++ {
			newPop.Set(i, k, pop.At(father, k))
		}
		for k := crossoverPoint; k < numQueens; k++ {
			newPop.Set(i, k, pop.At(mother, k))
		}

		// Mutation
		if rand.Float64() < mutationRate {
			mutationIndex := rand.Intn(numQueens)
			newValue := rand.Intn(numQueens)
			newPop.Set(i, mutationIndex, float64(newValue))
		}
	}

	return newPop
}

func getMaxInfo(v []float64) (int, float64) {
	n := len(v)
	currentMax := -1.0
	currentIndex := 0

	for i := 0; i < n; i++ {
		if v[i] > currentMax {
			currentMax = v[i]
			currentIndex = i
		}
	}

	return currentIndex, currentMax
}

func main() {
	popSize := 10
	numQueens := 8
	mutationRate := 0.05

	fmt.Println("Initializing...")

	pop := generatePopulation(popSize, numQueens)
	fit := fitnessVector(pop)
	mIdx, mVal := getMaxInfo(fit)
	gen := 0

	fmt.Println("Evolving...")

	for mVal < 0.99 {
		pop = crossover(pop, mutationRate, fit)
		fit = fitnessVector(pop)
		mIdx, mVal = getMaxInfo(fit)
		gen++
	}

	fmt.Println("Evolution ended.")
	fmt.Printf("Generation: %d\nFitness value:%f\n", gen, mVal)
	printMatrix(pop.RowView(mIdx), 3, 0)
}
