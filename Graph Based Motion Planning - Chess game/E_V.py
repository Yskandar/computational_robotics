import numpy as np

print("test")


chess_grid = dict()


count = 0
for i in range(8):
    for j in range(8):
        chess_grid[count] = [j,i]
        count+=1

print(chess_grid[0])

V = chess_grid

E = []


for i in range(64):
    if V[i][0] + 2 < 8:                 # +2 for x --> +/- 1 for y
        if V[i][1] + 1 < 8:
            next_state = i + 2 + 8
            E.append([i,next_state])
        if V[i][1] - 1 >= 0:
            next_state = i + 2 - 8
            E.append([i, next_state])

    if V[i][0] - 2 >= 0:              # - 2 for x --> +/- 1 for y
        if V[i][1] + 1 < 8:
            next_state = i - 2 + 8
            E.append([i, next_state])
        if V[i][1] - 1 >= 0:
            next_state = i - 2 - 8
            E.append([i, next_state])

    if V[i][1] + 2 < 8:                   # +2 in y --> +/- 1 in x
        if V[i][0] + 1 < 8:
            next_state = i + 16 + 1
            E.append([i, next_state])
        if V[i][0] - 1 >= 0:
            next_state = i + 16 - 1
            E.append([i, next_state])
    
    if V[i][1] - 2 >= 0:                  # -2 in y --> +/- 1 in x
        if V[i][0] + 1 < 8:
            next_state = i - 16 + 1
            E.append([i, next_state])
        if V[i][0] - 1 >= 0:
            next_state = i -16 - 1
            E.append([i, next_state])   

#print(E)
#print(len(E))


#BFS coding to find a way from initial state to final state


