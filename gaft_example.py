from math import sin, cos
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover,\
    FlipBitMutation

from gaft.analysis.fitness_store import FitnessStore

indv_template = BinaryIndividual(ranges=[(0, 10)], eps=0.0001)
population = Population(indv_template=indv_template, size=50)
population.init()

selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return x * sin(x) + x + cos(x)


if '__main__' == __name__:
    engine.run(ng=1000)
