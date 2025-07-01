#=============================================
# ALGORITMO GENETICO SIMPLE
# ALEX BRAULIO VON STERNENFELS HERNANDEZ 
# ESFM IPN FUNDAMENTOS DE IA
#=============================================
import datetime
import random

random.seed(random.random())
startTime = satetime.datetime.now()

# LOS GENES
geneSet = "abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

# OBJETIVO
target = "Alex Braulio Von"

# FRASE INICIAL
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))

# FUNCION DE APTITUD
def get_fitness(guess):
    return sum(1 for expected, actual in zip(target,guess) if expected == actual)

# MUTACION DE LETRAS EN LA FRASE
def mutate(parent):
    index = random.randrange(0,len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet,2))
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    return "".jpin(childGenes)

# MONITOREO DE LA SOLUCION
def display(guess):
    timeDiff = datetime.datetime.now() - startTime

