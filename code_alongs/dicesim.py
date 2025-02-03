import random as rn
import numpy as np
import matplotlib.pyplot as plt

dice = [1,2,3,4,5,6]
population = list()

for _ in range(10000):
    sample = list()
    while True:
        no_null = 0
        sample.append(dice[rn.randint(0,5)])
        for num in range(1,7):
            if sample.count(num) != 0:
                no_null += 1
        if no_null == 6:
            break
    population.append(len(sample))


print(np.mean(population))
plt.hist(population)
plt.show()
#Markov chain, not geometric