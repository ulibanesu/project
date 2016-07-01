#import numpy as np
import itertools

#def output(f1, f2, f3):
    
########################### FUNCTION TO CHECK F1 ##############################
def check_f1(X, f1):
    
    duplicates = [k for k, g in itertools.groupby(X) if len(list(g)) > 1]
    sign = [y - x for x,y in zip(duplicates, duplicates[1:])]
        
    for i in range(len(sign)):
        if (sign[i] > 0):
            f1.append(1)
        
        elif (sign[i] < 0):
            f1.append(-1)
    
    print(f1)

########################### FUNCTION TO CHECK F2 ##############################
def check_f2(X, f2):
    duplicates = [k for k, g in itertools.groupby(X) if len(list(g)) > 1]

    print("hello: ",duplicates)

    for i in range(len(duplicates)):
        
    
########################### FUNCTION TO CHECK F3 ##############################
#def check_f3(X, f3):

###############################################################################
########################### EXTRACT FROM THE DATA #############################
###############################################################################
filename1 = '/Users/KHATIBUSINESS/bitcoin/btceur_trade.txt'
        
with open(filename1) as f:
    tradeprice = f.read().splitlines()
        
for i in range(len(tradeprice)):
    tradeprice[i] = float(tradeprice[i])
#tradeprice = np.array(tradeprice)

print(tradeprice)

###################################### EXTRACT THE F1    
f1 = []

check_f1(tradeprice, f1)

###################################### EXTRACT THE F2    
f2 = []

check_f2(tradeprice, f2)


lst = [1,1,1,1,2,3,4,5,5,5,3,3,3,3,2,2,2,4,3]
duplicates = [k for k, g in itertools.groupby(lst) if len(list(g)) > 1] 
lol = [y - x for x,y in zip(duplicates, duplicates[1:])]
print(lol)