import numpy as np

belief = np.array([1/25]*25)

gridWorld = np.ones((5,5))
gridWorld[1,1] = 0
gridWorld[1,2] = 0
gridWorld[3,1] = 0
gridWorld[3,2] = 0



gridWorld /= 21

print(gridWorld)
print(belief)