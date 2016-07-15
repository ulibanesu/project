from __future__ import print_function

from itertools import zip_longest
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats


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
    

##################################### FUNCTION TO GET THE SIGN OF A NUMBER
def sign(number):
    """Will return 1 for positive,
    -1 for negative, and 0 for 0"""
    try:return number/abs(number)
    except ZeroDivisionError:return 0


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


##################################### CALCULATE THETA
def theta(phi, VWAP_Spread):
    
    features = []
    
    ###### Choice of alpha: (chose alpha=0.2) ######
    alpha = 0.20
    
    for i in range(len(phi)):
        first = abs((phi[i]/VWAP_Spread[i])/(phi[i-1]/VWAP_Spread[i-1]))
        if (first-1>alpha):
            first_out = 1
        
        elif (1-first>alpha):
            first_out = -1
            
        else:
            first_out = 0
        
        second = abs((phi[i]/VWAP_Spread[i])/(phi[i-2]/VWAP_Spread[i-2]))
        if (second-1>alpha):
            second_out = 1
        
        elif (1-second>alpha):
            second_out = -1
            
        else:
            second_out = 0
            
        third = abs((phi[i-1]/VWAP_Spread[i-1])/(phi[i-2]/VWAP_Spread[i-2]))
        if (third-1>alpha):
            third_out = 1
        
        elif (1-third>alpha):
            third_out = -1
            
        else:
            third_out = 0
  
        out = (first_out, second_out, third_out)
        features.append(out)
        
    return features
    
    
######################### FINAL FUNCTION TO GET THE F3 FEATURE
def final_output_f3(theta_output, phi, f3):

    for i in range(len(theta_output)):
        tuple = theta_output[i]
        first = tuple[0]
        second = tuple[1]
        third = tuple[2]
        
        if (first == 1 and second > -1 and third < 1 and (sign(phi[i]) == sign(phi[i-1]) or sign(phi[i]) == sign(phi[i-2]))):
            f3.append(1)

        elif (first == -1 and second < 1 and third > -1 and (sign(phi[i]) != sign(phi[i-1]) or sign(phi[i]) != sign(phi[i-2]))):
            f3.append(-1)
        
        else:
            f3.append(0)
    
    #print(f3)
    #print(len(f3))

###############################################################################
###############################################################################
########################### FUNCTION TO CHECK F1 ##############################

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

    ############ CALCULATE THETA
    theta_output = theta(phi, VWAP_Spread)
    
    ############ FINAL OUTPUT TO GET THE F3 FEATURE
    final_output_f3(theta_output, phi, f3)

###############################################################################
########################### EXTRACT FROM THE DATA #############################
###############################################################################

######## file containing the trade prices
filename1 = '/Users/KHATIBUSINESS/bitcoin/btceur_trade.txt'

######## file containing the trade prices dates
filename2 = '/Users/KHATIBUSINESS/bitcoin/btceur_date.txt'

####### file containing the order book dates
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

####################### MAKE THE FEATURE VECTOR ###############################
feature_vector = []

for i in range(len(f1)):
    first = f1[i]
    second = f2[i]
    third = f3[i]
    each_feature = (first, second, third)
    feature_vector.append(each_feature)

counter = 0
n = len(feature_vector)
s = 'feature_vector.txt'

with open(s,'w') as f:
    while (counter<n):
        f.write(str(feature_vector[counter])+'\n')
        counter += 1


############## FOR TEST PURPOSES
U1=0 
U2=0 
U3=0 
U4=0 
U5=0 
U6=0 
U7=0 
U8=0 
U9=0

D1=0        
D2=0
D3=0
D4=0
D5=0
D6=0
D7=0
D8=0
D9=0

for i in range(len(feature_vector)):
    
    ######### UP LEGS
    if (feature_vector[i] == (1,1,1)):
        U1+=1

    if (feature_vector[i] == (1,-1,1)):
        U2+=1

    if (feature_vector[i] == (1,1,0)):
        U3+=1

    if (feature_vector[i] == (1,0,1)):
        U4+=1

    if (feature_vector[i] == (1,0,0)):
        U5+=1

    if (feature_vector[i] == (1,0,-1)):
        U6+=1

    if (feature_vector[i] == (1,-1,0)):
        U7+=1

    if (feature_vector[i] == (1,1,-1)):
        U8+=1

    if (feature_vector[i] == (1,-1,-1)):
        U9+=1

    ######### DOWN LEGS
    if (feature_vector[i] == (-1,1,-1)):
        D1+=1

    if (feature_vector[i] == (-1,-1,-1)):
        D2+=1

    if (feature_vector[i] == (-1,1,0)):
        D3+=1

    if (feature_vector[i] == (-1,0,-1)):
        D4+=1

    if (feature_vector[i] == (-1,0,0)):
        D5+=1

    if (feature_vector[i] == (-1,0,1)):
        D6+=1

    if (feature_vector[i] == (-1,-1,0)):
        D7+=1

    if (feature_vector[i] == (-1,1,1)):
        D8+=1

    if (feature_vector[i] == (-1,-1,1)):
        D9+=1
        
        
print("U Trends:", U1,U2,U3,U4,U5,U6,U7,U8,U9)
print("D Trends:", D1,D2,D3,D4,D5,D6,D7,D8,D9)

distribution_zigzag_up = []
distribution_zigzag_up.append(U1) 
distribution_zigzag_up.append(U2)
distribution_zigzag_up.append(U3) 
distribution_zigzag_up.append(U4) 
distribution_zigzag_up.append(U5) 
distribution_zigzag_up.append(U6) 
distribution_zigzag_up.append(U7) 
distribution_zigzag_up.append(U8) 
distribution_zigzag_up.append(U9) 

distribution_zigzag_down = []
distribution_zigzag_down.append(D1) 
distribution_zigzag_down.append(D2)
distribution_zigzag_down.append(D3) 
distribution_zigzag_down.append(D4) 
distribution_zigzag_down.append(D5) 
distribution_zigzag_down.append(D6) 
distribution_zigzag_down.append(D7) 
distribution_zigzag_down.append(D8) 
distribution_zigzag_down.append(D9) 

#################### PLOT THE DISTRIBUTION (UP)

weights = distribution_zigzag_up/np.sum(distribution_zigzag_up)
x=[0,1,2,3,4,5,6,7,8]
labels=['U1','U2','U3','U4','U5','U6','U7','U8','U9']
plt.bar(x, weights, align='center')
plt.xticks(x, labels)
plt.xlabel("Distribution of zigzags")
plt.ylabel("Probability")

plt.show()

#################### PLOT THE DISTRIBUTION (DOWN)

weights = distribution_zigzag_down/np.sum(distribution_zigzag_down)
x=[0,1,2,3,4,5,6,7,8]
labels=['D1','D2','D3','D4','D5','D6','D7','D8','D9']
plt.bar(x, weights, align='center')
plt.xticks(x, labels)
plt.xlabel("Distribution of zigzags")
plt.ylabel("Probability")

plt.show()


###############################################################################
###############################################################################
############### MACHINE LEARNING PART (HIDDEN MARKOV MODEL) ###################

"""
Hidden Markov Model applied to BTC
--------------------------
Olivier Khatib
Imperial College London
2016
"""

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM


filename1 = '/Users/KHATIBUSINESS/bitcoin/btceur_trade.txt'

filename2 = '/Users/KHATIBUSINESS/bitcoin/btceur_volume.txt'

filename3 = '/Users/KHATIBUSINESS/bitcoin/btceur_date.txt'

print(__doc__)

###############################################################################

with open(filename1) as f:
    tradeprice = f.read().splitlines()

with open(filename2) as f:
    volume = f.read().splitlines()
    
with open(filename3) as f:
    dates = f.read().splitlines()

###################### RETRIEVE THE TRADE PRICE
for i in range(len(tradeprice)):
    tradeprice[i] = float(tradeprice[i])
tradeprice = np.array(tradeprice)
#tradeprice = tradeprice[1:]

###################### RETRIEVE THE VOLUME
for i in range(len(volume)):
    volume[i] = float(volume[i])
#volume = volume[1:]
volume = np.array(volume)

###################### RETRIEVE THE TRADE DATES
for i in range(len(dates)):
    dates[i] = int(dates[i])
#dates = dates[1:]
dates = np.array(dates)

data = []
for n in range(len(dates)):
    data.append(n)
data = np.array(data)

#diff = np.diff(tradeprice)

# Pack diff and volume for training.
X = np.column_stack([tradeprice])

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end="")
    
# Make an HMM instance and execute fit
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=2000, tol=0).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

print("Has the EM Algorithm converged?",model.monitor_.converged)

print(hidden_states)

###############################################################################
# Print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()

mean = []
var = []
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()
    mean.append(model.means_[i])
    var.append(np.diag(model.covars_[i]))


############# Retrieve the mean, variance and number of sample of each state
mean_state0 = mean[0] 
var_state0 = var[0]
   
mean_state1 = mean[1]
var_state1 = var[1]

mean_state2 = mean[2]  
var_state2 = var[2]  

no_state0 = 0
no_state1 = 0
no_state2 = 0

for i in range(len(hidden_states)):
    if (hidden_states[i] == 0):
        no_state0 += 1
    
    elif (hidden_states[i] == 1):
        no_state1 += 1

    elif (hidden_states[i] == 2):
        no_state2 += 1 
 
       
############ Plot the graph
       
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(data[mask], tradeprice[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()


#### CHECK IF A DISTRIBUTION OF ZIGZAGS IS MORE COMMON IN A HIDDEN STATE ######
########### STATE 0 is for REVERSAL
########### STATE 1 is for RUN
########### STATE 2 is for NEUTRAL

counter0 = 0
counter1 = 0
counter2 = 0
hidden_new = []

for i in range(len(boundaries)):
    tuple = boundaries[i]
    index_i = tuple[0]
    index_j = tuple[1]
    
    while (index_i <= index_j):
        if (hidden_states[index_i] == 0):
            counter0 += 1
        elif (hidden_states[index_i] == 1):
            counter1 += 1
        elif (hidden_states[index_i] == 2):
            counter2 += 1
        index_i +=1
        
    if (counter2 <= counter1 and counter0 < counter1):
        hidden_new.append(1)
        
    elif (counter0 <= counter2 and counter1 < counter2):
        hidden_new.append(2)
        
    else:
        hidden_new.append(0)
    
    counter0 = 0
    counter1 = 0
    counter2 = 0

#print(hidden_new)
U1_run=0 
U1_neu=0 
U1_rev=0
U2_run=0 
U2_neu=0 
U2_rev=0 
U3_run=0 
U3_neu=0 
U3_rev=0 
U4_run=0 
U4_neu=0 
U4_rev=0 
U5_run=0 
U5_neu=0 
U5_rev=0 
U6_run=0 
U6_neu=0 
U6_rev=0  
U7_run=0
U7_neu=0  
U7_rev=0  
U8_run=0
U8_neu=0  
U8_rev=0  
U9_run=0
U9_neu=0  
U9_rev=0 

D1_run=0 
D1_neu=0 
D1_rev=0
D2_run=0 
D2_neu=0 
D2_rev=0 
D3_run=0 
D3_neu=0 
D3_rev=0 
D4_run=0 
D4_neu=0 
D4_rev=0 
D5_run=0 
D5_neu=0 
D5_rev=0 
D6_run=0 
D6_neu=0 
D6_rev=0  
D7_run=0
D7_neu=0  
D7_rev=0  
D8_run=0
D8_neu=0  
D8_rev=0  
D9_run=0
D9_neu=0  
D9_rev=0

for i in range(len(hidden_new)):
    
    ### Case for up legs
    if (hidden_new[i] == 1 and feature_vector[i] == (1,1,1)):
        U1_run += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,1)):
        U1_neu += 1
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,1)):
        U1_rev += 1
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,1)):
        U2_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,1)):
        U2_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,1)):
        U2_rev += 1        
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,1,0)):
        U3_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,0)):
        U3_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,0)):
        U3_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,1)):
        U4_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,1)):
        U4_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,1)):
        U4_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,0)):
        U5_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,0)):
        U5_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,0)):
        U5_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,-1)):
        U6_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,-1)):
        U6_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,-1)):
        U6_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,0)):
        U7_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,0)):
        U7_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,0)):
        U7_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,1,-1)):
        U8_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,-1)):
        U8_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,-1)):
        U8_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,-1)):
        U9_run += 1   

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,-1)):
        U9_neu += 1   
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,-1)):
        U9_rev += 1            
        

    ### Case for down legs
    if (hidden_new[i] == 1 and feature_vector[i] == (-1,1,-1)):
        D1_run += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,-1)):
        D1_neu += 1
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,-1)):
        D1_rev += 1
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,-1)):
        D2_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,-1)):
        D2_neu += 2 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,-1)):
        D2_rev += 1        
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,1,0)):
        D3_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,0)):
        D3_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,0)):
        D3_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,-1)):
        D4_run += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,-1)):
        D4_neu += 1        
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,-1)):
        D4_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,0)):
        D5_run += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,0)):
        D5_neu += 1        
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,0)):
        D5_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,1)):
        D6_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,1)):
        D6_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,1)):
        D6_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,0)):
        D7_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,0)):
        D7_neu += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,0)):
        D7_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,1,1)):
        D8_run += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,1)):
        D8_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,1)):
        D8_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,1)):
        D9_run += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,1)):
        D9_neu += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,1)):
        D9_rev += 1         

        
print("U Trends:", U1_run,U2_run,U3_run,U4_run,U5_run,U6_run,U7_run,U8_run,U9_run)
print("U Trends:", U1_rev,U2_rev,U3_rev,U4_rev,U5_rev,U6_rev,U7_rev,U8_rev,U9_rev)
print("D Trends:", D1_run,D2_run,D3_run,D4_run,D5_run,D6_run,D7_run,D8_run,D9_run)
print("D Trends:", D1_rev,D2_rev,D3_rev,D4_rev,D5_rev,D6_rev,D7_rev,D8_rev,D9_rev)

distribution_zigzag_up_state = []
distribution_zigzag_up_state.append(U1_run) 
distribution_zigzag_up_state.append(U1_neu) 
distribution_zigzag_up_state.append(U1_rev)
distribution_zigzag_up_state.append(U2_run)
distribution_zigzag_up_state.append(U2_neu)  
distribution_zigzag_up_state.append(U2_rev)
distribution_zigzag_up_state.append(U3_run)
distribution_zigzag_up_state.append(U3_neu)  
distribution_zigzag_up_state.append(U3_rev)
distribution_zigzag_up_state.append(U4_run)
distribution_zigzag_up_state.append(U4_neu)  
distribution_zigzag_up_state.append(U4_rev)
distribution_zigzag_up_state.append(U5_run)
distribution_zigzag_up_state.append(U5_neu)  
distribution_zigzag_up_state.append(U5_rev)
distribution_zigzag_up_state.append(U6_run)
distribution_zigzag_up_state.append(U6_neu)  
distribution_zigzag_up_state.append(U6_rev)
distribution_zigzag_up_state.append(U7_run)
distribution_zigzag_up_state.append(U7_neu)  
distribution_zigzag_up_state.append(U7_rev)
distribution_zigzag_up_state.append(U8_run)
distribution_zigzag_up_state.append(U8_neu)  
distribution_zigzag_up_state.append(U8_rev)
distribution_zigzag_up_state.append(U9_run)
distribution_zigzag_up_state.append(U9_neu) 
distribution_zigzag_up_state.append(U9_rev)

distribution_zigzag_down_state = []
distribution_zigzag_down_state.append(D1_run) 
distribution_zigzag_down_state.append(D1_neu) 
distribution_zigzag_down_state.append(D1_rev)
distribution_zigzag_down_state.append(D2_run) 
distribution_zigzag_down_state.append(D2_neu) 
distribution_zigzag_down_state.append(D2_rev)
distribution_zigzag_down_state.append(D3_run)
distribution_zigzag_down_state.append(D3_neu)  
distribution_zigzag_down_state.append(D3_rev)
distribution_zigzag_down_state.append(D4_run)
distribution_zigzag_down_state.append(D4_neu)  
distribution_zigzag_down_state.append(D4_rev)
distribution_zigzag_down_state.append(D5_run) 
distribution_zigzag_down_state.append(D5_neu) 
distribution_zigzag_down_state.append(D5_rev)
distribution_zigzag_down_state.append(D6_run)
distribution_zigzag_down_state.append(D6_neu)  
distribution_zigzag_down_state.append(D6_rev)
distribution_zigzag_down_state.append(D7_run)
distribution_zigzag_down_state.append(D7_neu)  
distribution_zigzag_down_state.append(D7_rev)
distribution_zigzag_down_state.append(D8_run)
distribution_zigzag_down_state.append(D8_neu)  
distribution_zigzag_down_state.append(D8_rev)
distribution_zigzag_down_state.append(D9_run) 
distribution_zigzag_down_state.append(D9_neu) 
distribution_zigzag_down_state.append(D9_rev)

#################### PLOT THE DISTRIBUTION (UP)

weights = distribution_zigzag_up_state/np.sum(distribution_zigzag_up_state)
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
labels=['U1','U1','U1','U2','U2','U2','U3','U3','U3','U4','U4','U4',
        'U5','U5','U5','U6','U6','U6','U7','U7','U7','U8','U8','U8',
        'U9','U9','U9']
plt.bar(x, weights, width=0.7)
plt.xticks(x, labels)
plt.xlabel("Distribution of zigzags")
plt.ylabel("Probability")

plt.show()

#################### PLOT THE DISTRIBUTION (DOWN)

weights = distribution_zigzag_down_state/np.sum(distribution_zigzag_down_state)
x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
labels=['D1','D1','D1','D2','D2','D2','D3','D3','D3','D4','D4','D4',
        'D5','D5','D5','D6','D6','D6','D7','D7','D7','D8','D8','D8',
        'D9','D9','D9']
plt.bar(x, weights, width=0.7)
plt.xticks(x, labels)
plt.xlabel("Distribution of zigzags")
plt.ylabel("Probability")

plt.show()



###############################################################################
###############################################################################
############################ STATISTICAL SIGNIFICANCE #########################

#################### Welch's T-TEST (using a 5% significance level)
#################### H0: mean_state(i) == mean_state(j)
#################### H1: mean_state(i) != mean_state(j)
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr

######### 1st case: between state1 and state0
abar1 = mean_state1
avar1 = var_state1
na1 = no_state1
adof1 = na1 - 1

bbar1 = mean_state0
bvar1 = var_state0
nb1 = no_state0
bdof1 = nb1 - 1


tf1 = (abar1 - bbar1) / np.sqrt(avar1/na1 + bvar1/nb1)
dof1 = (avar1/na1 + bvar1/nb1)**2 / (avar1**2/(na1**2*adof1) + bvar1**2/(nb1**2*bdof1))
pf1 = 2*stdtr(dof1, -np.abs(tf1))

conf1_1 = (abar1 - bbar1)-tf1*np.sqrt(avar1/na1 + bvar1/nb1)
conf2_1 = (abar1 - bbar1)+tf1*np.sqrt(avar1/na1 + bvar1/nb1)

print(conf1_1, conf2_1)

if (abs(tf1)>pf1):
    print("Reject the null hypothesis (between State1 and State0)")
    
else:
    print("Cannot reject the null hypothesis (between State1 and State0)")
    
    
######### 2nd case: between state2 and state0
abar2 = mean_state2
avar2 = var_state2
na2 = no_state2
adof2 = na2 - 1

bbar2 = mean_state0
bvar2 = var_state0
nb2 = no_state0
bdof2 = nb2 - 1


tf2 = (abar2 - bbar2) / np.sqrt(avar2/na2 + bvar2/nb2)
dof2 = (avar2/na2 + bvar2/nb2)**2 / (avar2**2/(na2**2*adof2) + bvar2**2/(nb2**2*bdof2))
pf2 = 2*stdtr(dof2, -np.abs(tf2))

conf1_2 = (abar2 - bbar2)-tf2*np.sqrt(avar2/na2 + bvar2/nb2)
conf2_2 = (abar2 - bbar2)+tf2*np.sqrt(avar2/na2 + bvar2/nb2)

print(conf1_2, conf2_2)

if (abs(tf2)>pf2):
    print("Reject the null hypothesis (between State2 and State0)")
    
else:
    print("Cannot reject the null hypothesis (between State2 and State0)")    
    

######### 3rd case: between state2 and state1
abar3 = mean_state2
avar3 = var_state2
na3 = no_state2
adof3 = na3 - 1

bbar3 = mean_state1
bvar3 = var_state1
nb3 = no_state1
bdof3 = nb3 - 1


tf3 = (abar3 - bbar3) / np.sqrt(avar3/na3 + bvar3/nb3)
dof3 = (avar3/na3 + bvar3/nb3)**2 / (avar3**2/(na3**2*adof3) + bvar3**2/(nb3**2*bdof3))
pf3 = 2*stdtr(dof3, -np.abs(tf3))

conf1_3 = (abar2 - bbar2)-(tf2*np.sqrt(avar2/na2 + bvar2/nb2))
conf2_3 = (abar2 - bbar2)+(tf2*np.sqrt(avar2/na2 + bvar2/nb2))

print(conf1_3, conf2_3)

if (abs(tf3)>pf3):
    print("Reject the null hypothesis (between State2 and State1)")
    
else:
    print("Cannot reject the null hypothesis (between State2 and State1)") 



########## QQ-PLOT of the annualised returns
import numpy as np 
import pylab 
import scipy.stats as stats
from statistics import variance
from statistics import mean

list_state0 = []
returns_state0 = []
list_state1 = []
returns_state1 = []
list_state2 = []
returns_state2 = []

# Returns for Hidden States:
for i in range(len(hidden_states)):
    if (hidden_states[i] == 0):
        list_state0.append(tradeprice[i])

    elif (hidden_states[i] == 1):
        list_state1.append(tradeprice[i])
        
    elif (hidden_states[i] == 2):
        list_state2.append(tradeprice[i])
    

diff_state0 = np.diff(list_state0)
diff_state1 = np.diff(list_state1)
diff_state2 = np.diff(list_state2)

no_sample0 = len(diff_state0)
no_sample1 = len(diff_state1)
no_sample2 = len(diff_state2)

#### QQ-PLOT of Returns in Hidden State 0
stats.probplot(diff_state0, dist="norm", plot=pylab)
pylab.show()
print("Mean of Hidden State 0: ",mean(diff_state0))

#### QQ-PLOT of Returns in Hidden State 1
stats.probplot(diff_state1, dist="norm", plot=pylab)
pylab.show()
print("Mean of Hidden State 1: ",mean(diff_state1))

#### QQ-PLOT of Returns in Hidden State 2
stats.probplot(diff_state2, dist="norm", plot=pylab)
pylab.show()
print("Mean of Hidden State 2: ",mean(diff_state2))

    
###############################################################################
###############################################################################
############################### PREDICTION ####################################
