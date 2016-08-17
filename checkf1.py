# Helper function to go through parts
def gothrough(left, X, index):
    i = 1
    #output=0
    if (left):
        while (X[index-i] == X[index]):
            i+=1
        return index-i+1
    
    else: 
        while ((index+i)<len(X)) and (X[index+i] == X[index]): 
            i+=1
        return index+i-1

def check_f1(tradeprice, changes, f1, maxima, boundaries):
    
    #duplicates = [k for k, g in itertools.groupby(X) if len(list(g)) > 1]
    #sign = [y - x for x,y in zip(duplicates, duplicates[1:])]
    #f1 = []
    #maxima = []
    #changes = []
    #boundaries = []
    
    n = len(tradeprice)
    preV = 0
    currenT = 0
    nexT = 0
    index_maxima = 0
            
    while (currenT < n-1):
        index_maxima+=1
        currenT += 1
        
        if tradeprice[preV] == tradeprice[currenT]:
            preV += 1
            
        elif currenT == n-1:
            
            result = tradeprice[currenT]-tradeprice[preV]
            changes.append(result)
            maxima.append(tradeprice[currenT])
            first = gothrough(True, tradeprice, preV)
            second = gothrough(False, tradeprice, currenT)
            couple = (first, second)
            boundaries.append(couple)
        
        else: 
            nexT = currenT+1
            small = tradeprice[currenT]-tradeprice[preV]
            big = tradeprice[nexT] - tradeprice[preV]
                
            if not ((small < 0 and small > big) or (small > 0 and big > small)): 
                result = tradeprice[currenT] - tradeprice[preV]
                changes.append(result)
                maxima.append(tradeprice[preV])
                first = gothrough(True, tradeprice, preV)
                second = gothrough(False, tradeprice, currenT)
                couple = (first, second)
                boundaries.append(couple)
                preV = currenT
    
    for i in range(len(changes)):
        if (changes[i] > 0):
            f1.append(1)
        
        elif (changes[i] < 0):
            f1.append(-1)
    
    #print(f1)
    #print(len(f1))
    #print(boundaries)