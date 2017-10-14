import sys

sys.stdin = open('train.txt', 'r')
D = {}
D2 ={}
id = 0
#sum = sum_c = 0
for j in range(42068):
    line = sys.stdin.readline()[:-2]
    for c in line:
      #  sum_c += 1
        if c not in D2:
            D2[c] = id
            D[id] = c
            id += 1 
#print(sum)
#print(sum_c)
print(D2)
    
