#import numpy as np
#import itertools

#def output(f1, f2, f3):
    
########################### FUNCTION TO CHECK F1 ##############################
def check_f1(X, changes, f1, listnew):
    
    #duplicates = [k for k, g in itertools.groupby(X) if len(list(g)) > 1]
    #sign = [y - x for x,y in zip(duplicates, duplicates[1:])]
    
    n = len(X)
    preV = 0
    currenT = 0
    nexT = 0
            
    while (currenT < n-1):
        currenT += 1
        if X[preV] == X[currenT]:
            preV += 1
            
        elif currenT == n-1:
            result = X[currenT]-X[preV]
            changes.append(result)
            maxima.append(X[currenT])
        
        else: 
            nexT = currenT+1
            small = X[currenT]-X[preV]
            big = X[nexT] - X[preV]
                
            if not ((small < 0 and small > big) or (small > 0 and big > small)):                
                result = X[currenT] - X[preV]
                changes.append(result)
                maxima.append(X[preV])
                preV = currenT
    
    for i in range(len(changes)):
        if (changes[i] > 0):
            f1.append(1)
        
        elif (changes[i] < 0):
            f1.append(-1)
    
    print(f1)
    print(len(f1))

########################### FUNCTION TO CHECK F2 ##############################
def check_f2(X, f2, maxima):

    for i in range(len(maxima)):
        if i in range(len(maxima)-4):
            if ((maxima[i] < maxima[i+2] < maxima[i+4]) 
                and (maxima[i+1] < maxima[i+3])):
                    f2.append(1)
        
            elif ((maxima[i] > maxima[i+2] > maxima[i+4]) 
                and (maxima[i+1] > maxima[i+3])):
                    f2.append(-1)
        
            else: f2.append(0)
    
        else:
            if i == len(maxima)-4:
                if (maxima[i] < maxima[i+2] and maxima[i+1] < maxima[i+3]):
                    f2.append(1)
                
                elif (maxima[i] > maxima[i+2] and maxima[i+1] > maxima[i+3]):
                    f2.append(-1)
                
                else: f2.append(0)

            if i == len(maxima)-3:
                if (maxima[i] < maxima[i+2]):
                    f2.append(1)
                
                elif (maxima[i] > maxima[i+2]):
                    f2.append(-1)
                
                else: f2.append(0)  

            if i == len(maxima)-2:
                if (maxima[i] < maxima[i+1]):
                    f2.append(1)
                
                elif (maxima[i] > maxima[i+1]):
                    f2.append(-1)
                
                else: f2.append(0)

    print(f2)
    print(len(f2))    
    
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
maxima = []
changes = []

check_f1(tradeprice, changes, f1, maxima)

###################################### EXTRACT THE F2    
f2 = []

check_f2(tradeprice, f2, maxima)

