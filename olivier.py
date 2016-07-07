from itertools import zip_longest
from itertools import groupby
import numpy as np

################################### GET A SPECIFIC WORD IN THE LINE
def my_txt(text, target):
    count = 0
    last_was_space = False
    start = end = 0
    for index, letter in enumerate(text):
        if letter.isspace():
            if not last_was_space:
                 end = index
            last_was_space = True
        elif last_was_space:
            last_was_space = False
            count += 1
            if count > target:
                return text[start:end]
            elif count == target:
                start = index
    if count == target:
        return text[start:].strip()
    raise ValueError("Word not found")

#################################### GET RID OF THE EMPTY LINES
def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line
            
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

#################################### VWAP_bid Calculation    
def vwap_bid(list_of_chunk, output_bid):
    
    denominator = 0
    numerator = 0

    for member in list_of_chunk:
        bid_volume = float(my_txt(member,2))
        denominator += bid_volume

    for member in list_of_chunk:
        bid_price = float(my_txt(member,0))
        bid_volume = float(my_txt(member,2))
        numerator += bid_price*bid_volume
            
    result1 = numerator/denominator
    output_bid.append(result1)
    #print(output_bid)

#################################### VWAP_ask Calculation    
def vwap_ask(list_of_chunk, output_ask):
    
    denominator = 0
    numerator = 0

    for member in list_of_chunk:
        ask_volume = float(my_txt(member,8))
        denominator += ask_volume

    for member in list_of_chunk:
        ask_price = float(my_txt(member,6))
        ask_volume = float(my_txt(member,8))
        numerator += ask_price*ask_volume
    
    result2 = numerator/denominator
    output_ask.append(result2)

##################################### VWAP_bid_zigzag Calculation
def vwap_bid_zigzag(output_bid, output_bid_zigzag, 
                        boundaries, tradeprice_dates, orderbook_dates):

    # To be put when extracted
    output_bid = np.array(output_bid, dtype=float)
    output_bid_zigzag = np.array(output_bid_zigzag, dtype=float) 
    tradeprice_dates  = np.array(tradeprice_dates, dtype=int)
    orderbook_dates = np.array(orderbook_dates, dtype=int)


    numerator = np.zeros(len(boundaries))
    denominator = np.zeros(len(boundaries))
    last_time = orderbook_dates[len(orderbook_dates)-1]

    
    for n in range(len(boundaries)):
        
        # New version
        tuple = boundaries[n]
        time_trade_i = tradeprice_dates[int(tuple[0])]
        time_trade_j = tradeprice_dates[int(tuple[1])]

        if (time_trade_i > last_time):
            numerator[n] = 0
            denominator[n] = 0
            
        else:
            maskTmp = np.logical_and(orderbook_dates >= time_trade_i, orderbook_dates <= time_trade_j) 
            denominator[n] = sum(maskTmp)
            numerator[n] = sum(output_bid[maskTmp])
        
        result = numerator/denominator

    return result.tolist()
    #print(output_bid_zigzag)
    #print(result)
    #print(len(result))

##################################### VWAP_ask_zigzag Calculation
def vwap_ask_zigzag(output_ask, output_ask_zigzag, 
                        boundaries, tradeprice_dates, orderbook_dates):

    # To be put when extracted
    output_ask = np.array(output_ask, dtype=float)
    output_ask_zigzag = np.array(output_ask_zigzag, dtype=float) 
    tradeprice_dates  = np.array(tradeprice_dates, dtype=int)
    orderbook_dates = np.array(orderbook_dates, dtype=int)


    numerator = np.zeros(len(boundaries))
    denominator = np.zeros(len(boundaries))
    last_time = orderbook_dates[len(orderbook_dates)-1]

    
    for n in range(len(boundaries)):
        
        # New version
        #print(n)
        tuple = boundaries[n]
        time_trade_i = tradeprice_dates[int(tuple[0])]
        time_trade_j = tradeprice_dates[int(tuple[1])]

        if (time_trade_i > last_time):
            numerator[n] = 0
            denominator[n] = 0
            
        else:
            maskTmp = np.logical_and(orderbook_dates >= time_trade_i, orderbook_dates <= time_trade_j) 
            denominator[n] = sum(maskTmp)
            numerator[n] = sum(output_ask[maskTmp])
        
        result = numerator/denominator

    return result.tolist()


##################################### VWAP_bid_spread Calculation
def vwap_bid_spread(output_bid_zigzag, maxima, output_bid_spread):
    result = []    
    
    for i in range(len(maxima)):
        out = maxima[i]-output_bid_zigzag[i]
        result.append(out)
    return result
    

##################################### VWAP_bid_spread Calculation
def vwap_ask_spread(output_ask_zigzag, maxima, output_ask_spread):
    result = []    
    
    for i in range(len(maxima)):
        out = output_ask_zigzag[i]-maxima[i]
        result.append(out)
    return result


##################################### CALCULATE PHI
def phi_comp(output_bid_spread, output_ask_spread):
    result = []
    
    for i in range(len(output_bid_spread)):
        out = output_bid_spread[i]-output_ask_spread[i]
        result.append(out)
    return result


##################################### CALCULATE VWAP_Spread
def vwap_spread(output_bid_zigzag, output_ask_zigzag):
    result = []

    for i in range(len(output_bid_zigzag)):
        out = output_ask_zigzag[i]-output_bid_zigzag[i]
        result.append(out)
    return result

###############################################################################
###############################################################################
########################### FUNCTION TO CHECK F1 ##############################
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


def check_f1(X, changes, f1, listnew, boundaries):
    
    #duplicates = [k for k, g in itertools.groupby(X) if len(list(g)) > 1]
    #sign = [y - x for x,y in zip(duplicates, duplicates[1:])]
    
    n = len(X)
    preV = 0
    currenT = 0
    nexT = 0
    index_maxima = 0
            
    while (currenT < n-1):
        index_maxima+=1
        currenT += 1
        
        if X[preV] == X[currenT]:
            preV += 1
            
        elif currenT == n-1:
            
            result = X[currenT]-X[preV]
            changes.append(result)
            maxima.append(X[currenT])
            first = gothrough(True, X, preV)
            second = gothrough(False, X, currenT)
            couple = (first, second)
            boundaries.append(couple)
        
        else: 
            nexT = currenT+1
            small = X[currenT]-X[preV]
            big = X[nexT] - X[preV]
                
            if not ((small < 0 and small > big) or (small > 0 and big > small)): 
                result = X[currenT] - X[preV]
                changes.append(result)
                maxima.append(X[preV])
                first = gothrough(True, X, preV)
                second = gothrough(False, X, currenT)
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

###############################################################################
###############################################################################
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

            if i == len(maxima)-1:
                if (maxima[i] < maxima[i-1]):
                    f2.append(1)
                
                elif (maxima[i] > maxima[i-1]):
                    f2.append(-1)
                
                else: f2.append(0)

    #print(f2)
    #print(len(f2))    

###############################################################################
###############################################################################
########################### FUNCTION TO CHECK F3 ##############################

def check_f3(f3, boundaries, tradeprice_dates, orderbook_dates):

    with open('/Users/KHATIBUSINESS/bitcoin/test2.txt','r') as f:
        list_of_chunks = []
        output_bid = []
        output_ask = []
        output_bid_zigzag = []
        output_ask_zigzag = []
        output_bid_spread = []
        output_ask_spread = []

        for i, group_of_15 in groupby(enumerate(nonblank_lines(f)), key=lambda x: x[0]//15):
            chunk = list(map(lambda x: x[1], group_of_15))
            list_of_chunks.append(chunk)
            
        for list_of_chunk in list_of_chunks:    
            vwap_bid(list_of_chunk, output_bid)  
            vwap_ask(list_of_chunk, output_ask)
        
    f.close()

    
    ############ RETRIEVE output_bid_zigzag and output_ask_zigzag    
    output_bid_zigzag = vwap_bid_zigzag(output_bid, output_bid_zigzag, 
                                boundaries, tradeprice_dates, orderbook_dates)
    output_ask_zigzag = vwap_ask_zigzag(output_ask, output_ask_zigzag, 
                                boundaries, tradeprice_dates, orderbook_dates)

    ############ RETRIEVE output_bid_spread and output_ask_spread
    output_bid_spread = vwap_bid_spread(output_bid_zigzag, maxima, output_bid_spread)
    output_ask_spread = vwap_ask_spread(output_ask_zigzag, maxima, output_ask_spread)

    ############ CALCULATE PHI
    phi = phi_comp(output_bid_spread, output_ask_spread)

    ############ CALCULATE VWAP_Spread
    VWAP_Spread = vwap_spread(output_bid_zigzag, output_ask_zigzag)
    print(VWAP_Spread)


###############################################################################
########################### EXTRACT FROM THE DATA #############################
###############################################################################
filename1 = '/Users/KHATIBUSINESS/bitcoin/btceur_trade.txt'
filename2 = '/Users/KHATIBUSINESS/bitcoin/btceur_date.txt'
filename3 = '/Users/KHATIBUSINESS/bitcoin/btceur_orderbook_date.txt'

######### EXTRACT THE TRADE PRICE        
with open(filename1) as f:
    tradeprice = f.read().splitlines()
        
for i in range(len(tradeprice)):
    tradeprice[i] = float(tradeprice[i])

######### EXTRACT THE TRADE PRICE DATES        
with open(filename2) as f:
    tradeprice_dates = f.read().splitlines()
        
for i in range(len(tradeprice_dates)):
    tradeprice_dates[i] = int(tradeprice_dates[i])
    
######### EXTRACT THE ORDER BOOK DATES        
with open(filename3) as f:
    orderbook_dates = f.read().splitlines()
        
for i in range(len(tradeprice)):
    orderbook_dates[i] = int(orderbook_dates[i])

###################################### EXTRACT THE F1    
f1 = []
maxima = []
changes = []
boundaries = []

check_f1(tradeprice, changes, f1, maxima, boundaries)

###################################### EXTRACT THE F2    
f2 = []

check_f2(tradeprice, f2, maxima)

###################################### EXTRACT THE F3    
f3 = []

check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)

