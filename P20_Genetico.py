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
    newGene, alternate = 
