import numpy as np

print("test")


chess_grid = dict()


count = 0
for i in range(8):
    for j in range(8):
        chess_grid[count] = [i,j]
        count+=1

print(chess_grid[0])

V = chess_grid

E = {}

# Define a function that sets all the possible future states for current state and then we can search 
# the value inside the dictionary and if it exists then there is an edge. 


for i in range(63):
    if 0 <= V[i][0] + 2 <= 7:                          # x value
        if 0 <= V[i][1] + 1 <= 7:                       # Plus 2 in x and + 1 in y
            E[i] = [V[i],V[2*(i+1) + 1]]
    else:
        E[i] = 0
 

print(V)


# for i in range(63):
    
