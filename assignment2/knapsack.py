import time
import sys
sys.path.append("./ABAGAIL.jar")

from array import array
from time import clock
from itertools import product

from base import *

import java.util.Random as Random
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction

# Adapted from https://github.com/danielcy715/CS7641-Machine-Learning/blob/master/Assignment2/Knapsack.java

"""
Commandline parameter(s):
   none
"""

N = 50
maxIters = 3001
numTrials = 5
random = Random()

fill = [2] * N
ranges = array('i', fill)

MAX_VOLUME = 50
MAX_WEIGHT = 50
COPIES_EACH = 4

# values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# weights = [10, 5, 2, 8, 15, 11, 4, 7, 1, 20]
# maxWeight = 45

values = [0 for x in xrange(N)]
weights = [0 for x in xrange(N)]
# maxWeight = MAX_VOLUME * N * COPIES_EACH * 0.1
maxWeight = 100
copiesPerElement = [COPIES_EACH for x in xrange(N)]

for i in range(0, len(values)):
    values[i] = random.nextDouble() * MAX_VOLUME
    weights[i] = random.nextDouble() * MAX_WEIGHT
    # copiesPerElement[i] = random.nextInt(10) + 1

outfile = OUTPUT_DIRECTORY + '/KNAPSACK/KNAPSACK_{}_{}_LOG.csv'

# RHC
# for t in range(numTrials):
#     fname = outfile.format('RHC', str(t + 1))
#     with open(fname, 'w') as f:
#         f.write('iterations,fitness,time,fevals\n')
#
#     ef = KnapsackEvaluationFunction(values, weights, maxWeight, copiesPerElement)
#     odd = DiscreteUniformDistribution(ranges)           # DIFFERENT: DiscretePermutationDistribution(N)
#     nf = DiscreteChangeOneNeighbor(ranges)              # DIFFERENT: SwapNeighbor()
#     hcp = GenericHillClimbingProblem(ef, odd, nf)
#     rhc = RandomizedHillClimbing(hcp)
#     fit = FixedIterationTrainer(rhc, 10)
#
#     times = [0]
#     for i in range(0, maxIters, 10):
#         start = clock()
#         fit.train()
#         elapsed = time.clock() - start
#         times.append(times[-1] + elapsed)
#         score = ef.value(rhc.getOptimal())
#         st = '{},{},{},{}\n'.format(i, score, times[-1], 0)
#         print(st)
#         with open(fname, 'a') as f:
#             f.write(st)

# SA
for t in range(numTrials):
    for temperature, CE in product([1E5, 1E10], [0.15, 0.35, 0.55, 0.75, 0.95]):
        fname = outfile.format('SA{}'.format(CE), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        ef = KnapsackEvaluationFunction(values, weights, maxWeight, copiesPerElement)
        odd = DiscreteUniformDistribution(ranges)       # DIFFERENT: DiscretePermutationDistribution(N)
        nf = DiscreteChangeOneNeighbor(ranges)          # DIFFERENT: SwapNeighbor()
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(temperature, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)

        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(sa.getOptimal())
            st = '{},{},{},{}\n'.format(i, score, times[-1], 0)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# GA
# for t in range(numTrials):
#     for pop, mate, mutate in product([100, 200], [50, 30, 10], [50, 30, 10]):
#         fname = outfile.format('GA{}_{}_{}'.format(pop, mate, mutate), str(t + 1))
#         with open(fname, 'w') as f:
#             f.write('iterations,fitness,time,fevals\n')
#
#         ef = KnapsackEvaluationFunction(values, weights, maxWeight, copiesPerElement)
#         odd = DiscreteUniformDistribution(ranges)       # DIFFERENT: DiscretePermutationDistribution(N)
#         mf = DiscreteChangeOneMutation(ranges)          # DIFFERENT: SwapMutation()
#         cf = SingleCrossOver()                          # DIFFERENT: TravelingSalesmanCrossOver(ef)
#         gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
#         ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
#         fit = FixedIterationTrainer(ga, 10)
#
#         times = [0]
#         for i in range(0, maxIters, 10):
#             start = clock()
#             fit.train()
#             elapsed = time.clock() - start
#             times.append(times[-1] + elapsed)
#             score = ef.value(ga.getOptimal())
#             st = '{},{},{},{}\n'.format(i, score, times[-1], 0)
#             print(st)
#             with open(fname, 'a') as f:
#                 f.write(st)

# MIMIC
# for t in range(numTrials):
#     for samples, keep, m in product([100, 200], [10, 20, 50], [0.1, 0.3, 0.5, 0.7, 0.9]):
#         fname = outfile.format('MIMIC{}_{}_{}'.format(samples, keep, m), str(t + 1))
#         with open(fname, 'w') as f:
#             f.write('iterations,fitness,time,fevals\n')
#
#         ef = KnapsackEvaluationFunction(values, weights, maxWeight, copiesPerElement)
#         odd = DiscreteUniformDistribution(ranges)
#         df = DiscreteDependencyTree(m, ranges)
#         pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
#
#         mimic = MIMIC(samples, keep, pop)
#         fit = FixedIterationTrainer(mimic, 10)
#         times = [0]
#
#         for i in range(0, maxIters, 10):
#             start = clock()
#             fit.train()
#             elapsed = time.clock() - start
#             times.append(times[-1] + elapsed)
#             score = ef.value(mimic.getOptimal())
#             st = '{},{},{},{}\n'.format(i, score, times[-1], 0)
#             print(st)
#             with open(fname, 'a') as f:
#                 f.write(st)
