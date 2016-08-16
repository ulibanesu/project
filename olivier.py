from __future__ import print_function

from itertools import zip_longest
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
from scipy import stats as stat


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
    #print("and the winner is: ", len(output_bid_zigzag))
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
    
    return f3
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
    #f1 = []
    #maxima = []
    #changes = []
    #boundaries = []
    
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
    f3_ = final_output_f3(theta_output, phi, f3)

    return output_bid_zigzag, output_ask_zigzag, f3_

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

###################################### RETRIEVE THE F1    
f1 = []
maxima = []
changes = []
boundaries = []

check_f1(tradeprice, changes, f1, maxima, boundaries)

###################################### RETRIEVE THE F2    
f2 = []

check_f2(tradeprice, f2, maxima)

###################################### RETRIEVE THE F3    
f3 = []

check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)
output_bid_zigzag, output_ask_zigzag, f3_ = check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)


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

diff = []
diff.append(0)
diff.extend(np.diff(tradeprice))

diff = np.array(diff)

# Pack diff and volume for training.
X = np.column_stack([diff])

###############################################################################
###### Run the Gaussian Hidden Markov Model
print("Fitting the HMM ...", end="")
    
###### Make an HMM instance and fitting
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=2000, tol=0).fit(X)

###### Predict the optimal sequence of hidden states (using the Viterbi Algorithm)
hidden_states = []
#hidden_states.append(0)
hidden_states.extend(model.predict(X))

hidden_states = np.array(hidden_states)

print("The HMM has been processed")

print("Has the EM Algorithm converged? ",model.monitor_.converged)

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
    ax.plot_date(data[mask], diff[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()


#### CHECK IF A DISTRIBUTION OF ZIGZAGS IS MORE COMMON IN A HIDDEN STATE ######
########### STATE 0 is for REVERSAL (BEARISH)
########### STATE 1 is for NEUTRAL
########### STATE 2 is for RUN (BULLISH)

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
         U1_neu += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,1)):
         U1_run += 1
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,1)):
        U1_rev += 1
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,1)):
        U2_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,1)):
        U2_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,1)):
        U2_rev += 1        
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,1,0)):
        U3_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,0)):
        U3_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,0)):
        U3_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,1)):
        U4_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,1)):
        U4_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,1)):
        U4_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,0)):
        U5_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,0)):
        U5_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,0)):
        U5_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,0,-1)):
        U6_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,0,-1)):
        U6_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,0,-1)):
        U6_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,0)):
        U7_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,0)):
        U7_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,0)):
        U7_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,1,-1)):
        U8_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,1,-1)):
        U8_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,1,-1)):
        U8_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (1,-1,-1)):
        U9_neu += 1   

    elif (hidden_new[i] == 2 and feature_vector[i] == (1,-1,-1)):
        U9_run += 1   
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (1,-1,-1)):
        U9_rev += 1            
        

    ### Case for down legs
    if (hidden_new[i] == 1 and feature_vector[i] == (-1,1,-1)):
        D1_neu += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,-1)):
        D1_run += 1
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,-1)):
        D1_rev += 1
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,-1)):
        D2_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,-1)):
        D2_run += 2 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,-1)):
        D2_rev += 1        
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,1,0)):
        D3_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,0)):
        D3_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,0)):
        D3_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,-1)):
        D4_neu += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,-1)):
        D4_run += 1        
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,-1)):
        D4_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,0)):
        D5_neu += 1

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,0)):
        D5_run += 1        
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,0)):
        D5_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,0,1)):
        D6_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,0,1)):
        D6_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,0,1)):
        D6_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,0)):
        D7_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,0)):
        D7_run += 1  
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,0)):
        D7_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,1,1)):
        D8_neu += 1 

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,1,1)):
        D8_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,1,1)):
        D8_rev += 1            
        
    elif (hidden_new[i] == 1 and feature_vector[i] == (-1,-1,1)):
        D9_neu += 1  

    elif (hidden_new[i] == 2 and feature_vector[i] == (-1,-1,1)):
        D9_run += 1 
        
    elif (hidden_new[i] == 0 and feature_vector[i] == (-1,-1,1)):
        D9_rev += 1         

        
print("U Trends:", U1_run,U2_run,U3_run,U4_run,U5_run,U6_run,U7_run,U8_run,U9_run)
print("U Trends:", U1_neu,U2_neu,U3_neu,U4_neu,U5_neu,U6_neu,U7_neu,U8_neu,U9_neu)
print("U Trends:", U1_rev,U2_rev,U3_rev,U4_rev,U5_rev,U6_rev,U7_rev,U8_rev,U9_rev)

print("D Trends:", D1_run,D2_run,D3_run,D4_run,D5_run,D6_run,D7_run,D8_run,D9_run)
print("D Trends:", D1_neu,D2_neu,D3_neu,D4_neu,D5_neu,D6_neu,D7_neu,D8_neu,D9_neu)
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

conf1_1 = (abar1 - bbar1)-(tf1*np.sqrt(avar1/na1 + bvar1/nb1))
conf2_1 = (abar1 - bbar1)+(tf1*np.sqrt(avar1/na1 + bvar1/nb1))

print("p-value: ", pf1)
print("t-statistic: ", tf1)
print("Confidence interval: ", conf1_1, conf2_1)

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

conf1_2 = (abar2 - bbar2)-(tf2*np.sqrt(avar2/na2 + bvar2/nb2))
conf2_2 = (abar2 - bbar2)+(tf2*np.sqrt(avar2/na2 + bvar2/nb2))

print("p-value: ", pf2)
print("t-statistic: ", tf2)
print("Confidence interval: ", conf1_2, conf2_2)

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

conf1_3 = (abar3 - bbar3)-(tf3*np.sqrt(avar3/na3 + bvar3/nb3))
conf2_3 = (abar3 - bbar3)+(tf3*np.sqrt(avar3/na3 + bvar3/nb3))

print("p-value: ", pf3)
print("t-statistic: ", tf3)
print("Confidence interval: ", conf1_3, conf2_3)

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
print("Mean of returns of Hidden State 0: ",mean(diff_state0))
print("Variance of returns of Hidden State 0: ",variance(diff_state0))

#### QQ-PLOT of Returns in Hidden State 1
stats.probplot(diff_state1, dist="norm", plot=pylab)
pylab.show()
print("Mean of returns of Hidden State 1: ",mean(diff_state1))
print("Variance of returns of Hidden State 1: ",variance(diff_state1))

#### QQ-PLOT of Returns in Hidden State 2
stats.probplot(diff_state2, dist="norm", plot=pylab)
pylab.show()
print("Mean of returns of Hidden State 2: ",mean(diff_state2))
print("Variance of returns of Hidden State 2: ",variance(diff_state2))


######################### Returns from Hidden State 0
mu, sigma = mean(diff_state0), variance(diff_state0)
x = diff_state0




# the histogram of the data
plt.figure(1)
histvals, binvals, patches = plt.hist(
   x, bins=50, normed=1, facecolor='g', alpha=0.75, label='my data')

pdf = stat.norm.pdf(binvals)
plt.plot(binvals, pdf, 'r--')

plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Gaussian distribution of Hidden State 0')
plt.text(-2, 0.45, r'$\mu=0,\ \sigma=1$')
plt.xlim(-1.5, 1.5)
plt.ylim(0, 8)
plt.grid(True)


plt.show()
######################### Returns from Hidden State 1
mu, sigma = mean(diff_state1), variance(diff_state1)
x = diff_state1


plt.figure(2)
histvals, binvals, patches = plt.hist(
   x, bins=50, normed=1, facecolor='g', alpha=0.75, label='my data')

pdf = stat.norm.pdf(binvals)
plt.plot(binvals, pdf, 'r--')

plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Gaussian distribution of Hidden State 1')
plt.text(-2, 0.45, r'$\mu=0,\ \sigma=1$')
plt.xlim(-1.5, 1.5)
plt.ylim(0, 8)
plt.grid(True)

######################### Returns from Hidden State 2
mu, sigma = mean(diff_state2), variance(diff_state2)
x = diff_state2
plt.figure(3)
histvals, binvals, patches = plt.hist(
   x, bins=50, normed=1, facecolor='g', alpha=0.75, label='my data')

pdf = stat.norm.pdf(binvals)
plt.plot(binvals, pdf, 'r--')

plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Gaussian distribution of Hidden State 2')
plt.text(-2, 0.45, r'$\mu=0,\ \sigma=1$')
plt.xlim(-1.5, 1.5)
plt.ylim(0, 8)
plt.grid(True)


    
###############################################################################
###############################################################################
############################### PREDICTION ####################################

###############################################################################
#################### GET THE DISTRIBUTION OF EACH FEATURE VECTOR ##############
########################## IN EACH OF THE HIDDEN STATES #######################

#### CASE FOR U1
if (U1_run + U1_neu + U1_rev == 0):
    U1_run_prob = 0
    U1_neu_prob = 0
    U1_rev_prob = 0
else:
    U1_run_prob = U1_run/(U1_run + U1_neu + U1_rev)
    U1_neu_prob = U1_neu/(U1_run + U1_neu + U1_rev)
    U1_rev_prob = U1_rev/(U1_run + U1_neu + U1_rev)

#### CASE FOR U2
if (U2_run + U2_neu + U2_rev == 0):
    U2_run_prob = 0
    U2_neu_prob = 0
    U2_rev_prob = 0
else:
    U2_run_prob = U2_run/(U2_run + U2_neu + U2_rev)
    U2_neu_prob = U2_neu/(U2_run + U2_neu + U2_rev)
    U2_rev_prob = U2_rev/(U2_run + U2_neu + U2_rev)

#### CASE FOR U3
if (U3_run + U3_neu + U3_rev == 0):
    U3_run_prob = 0
    U3_neu_prob = 0
    U3_rev_prob = 0
else:
    U3_run_prob = U3_run/(U3_run + U3_neu + U3_rev)
    U3_neu_prob = U3_neu/(U3_run + U3_neu + U3_rev)
    U3_rev_prob = U3_rev/(U3_run + U3_neu + U3_rev)

#### CASE FOR U4
U4_run_prob = U4_run/(U4_run + U4_neu + U4_rev)
U4_neu_prob = U4_neu/(U4_run + U4_neu + U4_rev)
U4_rev_prob = U4_rev/(U4_run + U4_neu + U4_rev)

#### CASE FOR U5
U5_run_prob = U5_run/(U5_run + U5_neu + U5_rev)
U5_neu_prob = U5_neu/(U5_run + U5_neu + U5_rev)
U5_rev_prob = U5_rev/(U5_run + U5_neu + U5_rev)

#### CASE FOR U6
U6_run_prob = U6_run/(U6_run + U6_neu + U6_rev)
U6_neu_prob = U6_neu/(U6_run + U6_neu + U6_rev)
U6_rev_prob = U6_rev/(U6_run + U6_neu + U6_rev)

#### CASE FOR U7
if (U7_run + U7_neu + U7_rev == 0):
    U7_run_prob = 0
    U7_neu_prob = 0
    U7_rev_prob = 0
else:
    U7_run_prob = U7_run/(U7_run + U7_neu + U7_rev)
    U7_neu_prob = U7_neu/(U7_run + U7_neu + U7_rev)
    U7_rev_prob = U7_rev/(U7_run + U7_neu + U7_rev)

#### CASE FOR U8
if (U8_run + U8_neu + U8_rev == 0):
    U8_run_prob = 0
    U8_neu_prob = 0
    U8_rev_prob = 0
else:
    U8_run_prob = U8_run/(U8_run + U8_neu + U8_rev)
    U8_neu_prob = U8_neu/(U8_run + U8_neu + U8_rev)
    U8_rev_prob = U8_rev/(U8_run + U8_neu + U8_rev)

#### CASE FOR U9
if (U9_run + U9_neu + U9_rev == 0):
    U9_run_prob = 0
    U9_neu_prob = 0
    U9_rev_prob = 0
else:
    U9_run_prob = U9_run/(U9_run + U9_neu + U9_rev)
    U9_neu_prob = U9_neu/(U9_run + U9_neu + U9_rev)
    U9_rev_prob = U9_rev/(U9_run + U9_neu + U9_rev)

#### CASE FOR D1
if (D1_run + D1_neu + D1_rev == 0):
    D1_run_prob = 0
    D1_neu_prob = 0
    D1_rev_prob = 0
else:
    D1_run_prob = D1_run/(D1_run + D1_neu + D1_rev)
    D1_neu_prob = D1_neu/(D1_run + D1_neu + D1_rev)
    D1_rev_prob = D1_rev/(D1_run + D1_neu + D1_rev)

#### CASE FOR D2
if (D2_run + D2_neu + D2_rev == 0):
    D2_run_prob = 0
    D2_neu_prob = 0
    D2_rev_prob = 0
else:
    D2_run_prob = D2_run/(D2_run + D2_neu + D2_rev)
    D2_neu_prob = D2_neu/(D2_run + D2_neu + D2_rev)
    D2_rev_prob = D2_rev/(D2_run + D2_neu + D2_rev)

#### CASE FOR D3
if (D3_run + D3_neu + D3_rev == 0):
    D3_run_prob = 0
    D3_neu_prob = 0
    D3_rev_prob = 0
else:
    D3_run_prob = D3_run/(D3_run + D3_neu + D3_rev)
    D3_neu_prob = D3_neu/(D3_run + D3_neu + D3_rev)
    D3_rev_prob = D3_rev/(D3_run + D3_neu + D3_rev)

#### CASE FOR D4
D4_run_prob = D4_run/(D4_run + D4_neu + D4_rev)
D4_neu_prob = D4_neu/(D4_run + D4_neu + D4_rev)
D4_rev_prob = D4_rev/(D4_run + D4_neu + D4_rev)

#### CASE FOR D5
D5_run_prob = D5_run/(D5_run + D5_neu + D5_rev)
D5_neu_prob = D5_neu/(D5_run + D5_neu + D5_rev)
D5_rev_prob = D5_rev/(D5_run + D5_neu + D5_rev)

#### CASE FOR D6
D6_run_prob = D6_run/(D6_run + D6_neu + D6_rev)
D6_neu_prob = D6_neu/(D6_run + D6_neu + D6_rev)
D6_rev_prob = D6_rev/(D6_run + D6_neu + D6_rev)

#### CASE FOR D7
if (D7_run + D7_neu + D7_rev == 0):
    D7_run_prob = 0
    D7_neu_prob = 0
    D7_rev_prob = 0
else:
    D7_run_prob = D7_run/(D7_run + D7_neu + D7_rev)
    D7_neu_prob = D7_neu/(D7_run + D7_neu + D7_rev)
    D7_rev_prob = D7_rev/(D7_run + D7_neu + D7_rev)

#### CASE FOR D8
if (D8_run + D8_neu + D8_rev == 0):
    D8_run_prob = 0
    D8_neu_prob = 0
    D8_rev_prob = 0
else:
    D8_run_prob = D8_run/(D8_run + D8_neu + D8_rev)
    D8_neu_prob = D8_neu/(D8_run + D8_neu + D8_rev)
    D8_rev_prob = D8_rev/(D8_run + D8_neu + D8_rev)

#### CASE FOR D9
if (D9_run + D9_neu + D9_rev == 0):
    D9_run_prob = 0
    D9_neu_prob = 0
    D9_rev_prob = 0
else:
    D9_run_prob = D9_run/(D9_run + D9_neu + D9_rev)
    D9_neu_prob = D9_neu/(D9_run + D9_neu + D9_rev)
    D9_rev_prob = D9_rev/(D9_run + D9_neu + D9_rev)


###############################################################################
#################### GET THE DISTRIBUTION OF FEATURE VECTORS ##################
########################## IN EACH OF THE HIDDEN STATES #######################

###### CASE FOR HIDDEN STATE 0 (REVERSAL)
state_rev_prob = []
state_rev_denom = U1_rev + U2_rev + U3_rev + U4_rev + U5_rev + U6_rev + U7_rev + U8_rev + U9_rev + D1_rev + D2_rev + D3_rev + D4_rev + D5_rev + D6_rev + D7_rev + D8_rev + D9_rev

state_rev_prob.append(U1_rev/state_rev_denom)
state_rev_prob.append(U2_rev/state_rev_denom)
state_rev_prob.append(U3_rev/state_rev_denom)
state_rev_prob.append(U4_rev/state_rev_denom)
state_rev_prob.append(U5_rev/state_rev_denom)
state_rev_prob.append(U6_rev/state_rev_denom)
state_rev_prob.append(U7_rev/state_rev_denom)
state_rev_prob.append(U8_rev/state_rev_denom)
state_rev_prob.append(U9_rev/state_rev_denom)
state_rev_prob.append(D1_rev/state_rev_denom)
state_rev_prob.append(D2_rev/state_rev_denom)
state_rev_prob.append(D3_rev/state_rev_denom)
state_rev_prob.append(D4_rev/state_rev_denom)
state_rev_prob.append(D5_rev/state_rev_denom)
state_rev_prob.append(D6_rev/state_rev_denom)
state_rev_prob.append(D7_rev/state_rev_denom)
state_rev_prob.append(D8_rev/state_rev_denom)
state_rev_prob.append(D9_rev/state_rev_denom)

###### CASE FOR HIDDEN STATE 1 (NEUTRAL)
state_neu_prob = []
state_neu_denom = U1_neu + U2_neu + U3_neu + U4_neu + U5_neu + U6_neu + U7_neu + U8_neu + U9_neu + D1_neu + D2_neu + D3_neu + D4_neu + D5_neu + D6_neu + D7_neu + D8_neu + D9_neu

state_neu_prob.append(U1_neu/state_neu_denom)
state_neu_prob.append(U2_neu/state_neu_denom)
state_neu_prob.append(U3_neu/state_neu_denom)
state_neu_prob.append(U4_neu/state_neu_denom)
state_neu_prob.append(U5_neu/state_neu_denom)
state_neu_prob.append(U6_neu/state_neu_denom)
state_neu_prob.append(U7_neu/state_neu_denom)
state_neu_prob.append(U8_neu/state_neu_denom)
state_neu_prob.append(U9_neu/state_neu_denom)
state_neu_prob.append(D1_neu/state_neu_denom)
state_neu_prob.append(D2_neu/state_neu_denom)
state_neu_prob.append(D3_neu/state_neu_denom)
state_neu_prob.append(D4_neu/state_neu_denom)
state_neu_prob.append(D5_neu/state_neu_denom)
state_neu_prob.append(D6_neu/state_neu_denom)
state_neu_prob.append(D7_neu/state_neu_denom)
state_neu_prob.append(D8_neu/state_neu_denom)
state_neu_prob.append(D9_neu/state_neu_denom)

###### CASE FOR HIDDEN STATE 2 (RUN)
state_run_prob = []
state_run_denom = U1_run + U2_run + U3_run + U4_run + U5_run + U6_run + U7_run + U8_run + U9_run + D1_run + D2_run + D3_run + D4_run + D5_run + D6_run + D7_run + D8_run + D9_run

state_run_prob.append(U1_run/state_run_denom)
state_run_prob.append(U2_run/state_run_denom)
state_run_prob.append(U3_run/state_run_denom)
state_run_prob.append(U4_run/state_run_denom)
state_run_prob.append(U5_run/state_run_denom)
state_run_prob.append(U6_run/state_run_denom)
state_run_prob.append(U7_run/state_run_denom)
state_run_prob.append(U8_run/state_run_denom)
state_run_prob.append(U9_run/state_run_denom)
state_run_prob.append(D1_run/state_run_denom)
state_run_prob.append(D2_run/state_run_denom)
state_run_prob.append(D3_run/state_run_denom)
state_run_prob.append(D4_run/state_run_denom)
state_run_prob.append(D5_run/state_run_denom)
state_run_prob.append(D6_run/state_run_denom)
state_run_prob.append(D7_run/state_run_denom)
state_run_prob.append(D8_run/state_run_denom)
state_run_prob.append(D9_run/state_run_denom)

###############################################################################
###############################################################################
############################# MODEL SELECTION #################################
### AIC
temp = model.decode(X)
logprob = temp[0]
k1 = 3^2 + 2*3 - 1  # Case with 3 hidden states
k2 = 2^2 + 2*2 - 1  # Case with 2 hidden states
k3 = 4^2 + 2*4 - 1  # Case with 4 hidden states
k4 = 5^2 + 2*5 - 1  # Case with 5 hidden states

AIC1 = -2*logprob + 2*k1
AIC2 = -2*logprob + 2*k2
AIC3 = -2*logprob + 2*k3
AIC4 = -2*logprob + 2*k4

### BIC
import math
logM = math.log(len(tradeprice))

BIC1 = -2*logprob + k1*logM
BIC2 = -2*logprob + k2*logM
BIC3 = -2*logprob + k3*logM
BIC4 = -2*logprob + k4*logM



###############################################################################
###############################################################################
################################ PREDICTION ###################################
from random import random

state_time_t = hidden_states[-1]    # Get the latest hidden state
predicted_states = []

###############################################################################
###############################################################################
############# RETRIEVE THE LAST feature_vector (ie. the one at time t) ########

last_feature_vector = feature_vector[-1]    # Get the latest feature vector

if (state_time_t == 0):
    
    rand = random()
    a00 = (model.transmat_[0][0])
    a01 = (model.transmat_[0][1])
    a02 = (model.transmat_[0][2])        

    if (last_feature_vector == (1,1,1)):
        a02 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a02 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a02 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a02 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a02 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a02 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a02 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a02 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a01 += 0.1

    elif (last_feature_vector == (-1,0,0)):
        a01 += 0.1

    elif (last_feature_vector == (1,0,-1)):
        a00 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a00 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a00 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a00 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a00 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a00 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a00 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a00 += 0.5
    
    denom = (a00 + a01 + a02) 
    a00 = a00/denom    
    a01 = a01/denom    
    a02 = a02/denom 
    
    distrib0 = a00
    distrib1 = a00 + a01
    distrib2 = a00 + a01 + a02
    
    #### goes to the next time (ie. t+1)
    if (rand <= distrib0):
        state_time_tPlusOne = 0
    
    elif (rand <= distrib1):
        state_time_tPlusOne = 1
    
    elif (rand <= distrib2):
        state_time_tPlusOne = 2

elif (state_time_t == 1):

    rand = random()
    a10 = (model.transmat_[1][0])
    a11 = (model.transmat_[1][1])
    a12 = (model.transmat_[1][2])

    if (last_feature_vector == (1,1,1)):
        a12 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a12 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a12 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a12 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a12 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a12 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a12 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a12 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a11 += 0.1

    elif (last_feature_vector == (-1,0,0)):
        a11 += 0.1

    elif (last_feature_vector == (1,0,-1)):
        a10 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a10 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a10 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a10 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a10 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a10 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a10 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a10 += 0.5
    
    denom = (a10 + a11 + a12) 
    a10 = a10/denom    
    a11 = a11/denom
    a12 = a12/denom 
    
    distrib0 = a10
    distrib1 = a10 + a11
    distrib2 = a10 + a11 + a12

    #### goes to the next time (ie. t+1)    
    if (rand <= distrib0):
        state_time_tPlusOne = 0
    
    elif (rand <= distrib1):
        state_time_tPlusOne = 1
    
    elif (rand <= distrib2):
        state_time_tPlusOne = 2    

elif (state_time_t == 2):
    
    rand = random()
    a20 = (model.transmat_[2][0])
    a21 = (model.transmat_[2][1])
    a22 = (model.transmat_[2][2])
    
    if (last_feature_vector == (1,1,1)):
        a22 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a22 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a22 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a22 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a22 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a22 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a22 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a22 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a21 += 0.1

    elif (last_feature_vector == (-1,0,0)):
        a21 += 0.1

    elif (last_feature_vector == (1,0,-1)):
        a20 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a20 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a20 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a20 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a20 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a20 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a20 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a20 += 0.5
    
    denom = (a20 + a21 + a22)
    a20 = a20/denom 
    a21 = a21/denom    
    a22 = a22/denom   
    
    distrib0 = a20
    distrib1 = a20 + a21
    distrib2 = a20 + a21 + a22

    #### goes to the next time (ie. t+1)    
    if (rand <= distrib0):
        state_time_tPlusOne = 0
    
    elif (rand <= distrib1):
        state_time_tPlusOne = 1
    
    elif (rand <= distrib2):
        state_time_tPlusOne = 2 

predicted_states.append(state_time_tPlusOne)


###############################################################################
###############################################################################
############################# TIME (T+1) ######################################
return_state0 = model.means_[0]
return_state1 = model.means_[1]
return_state2 = model.means_[2]

current_price = tradeprice[-1]

if (predicted_states[-1] == 0):

    """ 
    Need to think about this bruv
        - append the new computed price at that time
        - take that price into account to compute the new zigzag
        - think about the hidden state at time t+1
        - think about the probs of going to another state according to the 
            COMPUTED FEATURE VECTOR and the transition matrix 
    """

    tradeprice.append(current_price+return_state0)

    current_price = tradeprice[-1]
    
    ###################################### RETRIEVE THE F1    
    f1 = []
    maxima = []
    changes = []
    boundaries = []

    check_f1(tradeprice, changes, f1, maxima, boundaries)
    
    ###################################### RETRIEVE THE F2    
    f2 = []
    
    check_f2(tradeprice, f2, maxima)
    
    ###################################### RETRIEVE THE F3    
    f3 = []
    
    check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)

    first = f1[-1]
    second = f2[-1]
    third = f3[-1]
    each_feature = (first, second, third)
    feature_vector.append(each_feature)
    
    last_feature_vector = feature_vector[-1]     # RETRIEVE THE LAST FEATURE VECTOR
    
    
    rand = random()
    a00 = (model.transmat_[0][0])
    a01 = (model.transmat_[0][1])
    a02 = (model.transmat_[0][2])        

    if (last_feature_vector == (1,1,1)):
        a02 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a02 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a02 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a02 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a02 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a02 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a02 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a02 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a01 += 0

    elif (last_feature_vector == (-1,0,0)):
        a01 += 0

    elif (last_feature_vector == (1,0,-1)):
        a00 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a00 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a00 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a00 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a00 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a00 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a00 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a00 += 0.5


    denom = (a00 + a01 + a02)
    a00 = a00/denom 
    a01 = a01/denom    
    a02 = a02/denom 
    
    distrib0 = a02/(a02 + a00)
    distrib1 = a00/(a02 + a00)
    total0 = distrib0
    total1 = distrib0 + distrib1
    
    #### goes to the next time (ie. t+2)
    if (rand <= total0):
        predicted_states.append(2)
    
    elif (rand <= total1):
        predicted_states.append(0)



elif (predicted_states[-1] == 1):

    tradeprice.append(current_price+return_state1)

    current_price = tradeprice[-1]
    
    ###################################### RETRIEVE THE F1    
    f1 = []
    maxima = []
    changes = []
    boundaries = []

    check_f1(tradeprice, changes, f1, maxima, boundaries)
    
    ###################################### RETRIEVE THE F2    
    f2 = []
    
    check_f2(tradeprice, f2, maxima)
    
    ###################################### RETRIEVE THE F3    
    f3 = []
    
    check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)

    first = f1[-1]
    second = f2[-1]
    third = f3[-1]
    each_feature = (first, second, third)
    feature_vector.append(each_feature)
    
    last_feature_vector = feature_vector[-1]     # RETRIEVE THE LAST FEATURE VECTOR
    
    
    rand = random()
    a10 = (model.transmat_[1][0])
    a11 = (model.transmat_[1][1])
    a12 = (model.transmat_[1][2])        

    if (last_feature_vector == (1,1,1)):
        a12 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a12 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a12 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a12 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a12 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a12 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a12 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a12 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a11 += 0

    elif (last_feature_vector == (-1,0,0)):
        a11 += 0

    elif (last_feature_vector == (1,0,-1)):
        a10 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a10 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a10 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a10 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a10 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a10 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a10 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a10 += 0.5


    denom = (a10 + a11 + a12)
    a10 = a10/denom 
    a11 = a11/denom    
    a12 = a12/denom 
    
    distrib0 = a12/(a12 + a10)
    distrib1 = a10/(a12 + a10)
    total0 = distrib0
    total1 = distrib0 + distrib1
    
    #### goes to the next time (ie. t+2)
    if (rand <= total0):
        predicted_states.append(2)
    
    elif (rand <= total1):
        predicted_states.append(0)

elif (predicted_states[-1] == 2):

    tradeprice.append(current_price+return_state2)

    current_price = tradeprice[-1]
    
    ###################################### RETRIEVE THE F1    
    f1 = []
    maxima = []
    changes = []
    boundaries = []

    check_f1(tradeprice, changes, f1, maxima, boundaries)
    
    ###################################### RETRIEVE THE F2    
    f2 = []
    
    check_f2(tradeprice, f2, maxima)
    
    ###################################### RETRIEVE THE F3    
    f3 = []
    
    check_f3(f3, boundaries, tradeprice_dates, orderbook_dates)

    first = f1[-1]
    second = f2[-1]
    third = f3[-1]
    each_feature = (first, second, third)
    feature_vector.append(each_feature)
    
    last_feature_vector = feature_vector[-1]     # RETRIEVE THE LAST FEATURE VECTOR
    
    
    rand = random()
    a20 = (model.transmat_[2][0])
    a21 = (model.transmat_[2][1])
    a22 = (model.transmat_[2][2])        

    if (last_feature_vector == (1,1,1)):
        a22 += 0.5

    elif (last_feature_vector == (-1,1,-1)):
        a22 += 0.5
    
    elif (last_feature_vector == (1,-1,1)):
        a22 += 0.4

    elif (last_feature_vector == (-1,-1,-1)):
        a22 += 0.4
    
    elif (last_feature_vector == (1,1,0)):
        a22 += 0.3

    elif (last_feature_vector == (-1,1,0)):
        a22 += 0.3

    elif (last_feature_vector == (1,0,1)):
        a22 += 0.2

    elif (last_feature_vector == (-1,0,-1)):
        a22 += 0.2

    elif (last_feature_vector == (1,0,0)):
        a21 += 0

    elif (last_feature_vector == (-1,0,0)):
        a21 += 0

    elif (last_feature_vector == (1,0,-1)):
        a20 += 0.2

    elif (last_feature_vector == (-1,0,1)):
        a20 += 0.2

    elif (last_feature_vector == (1,-1,0)):
        a20 += 0.3

    elif (last_feature_vector == (-1,-1,0)):
        a20 += 0.3

    elif (last_feature_vector == (1,1,-1)):
        a20 += 0.4

    elif (last_feature_vector == (-1,1,1)):
        a20 += 0.4

    elif (last_feature_vector == (1,-1,-1)):
        a20 += 0.5

    elif (last_feature_vector == (-1,-1,1)):
        a20 += 0.5


    denom = (a20 + a21 + a22)
    a20 = a20/denom 
    a21 = a21/denom    
    a22 = a22/denom 
    
    distrib0 = a22/(a22 + a20)
    distrib1 = a20/(a22 + a20)
    total0 = distrib0
    total1 = distrib0 + distrib1
    
    #### goes to the next time (ie. t+2)
    if (rand <= total0):
        predicted_states.append(2)
    
    elif (rand <= total1):
        predicted_states.append(0)











"""
if (predicted_states[-1] == 0):
    predicted_states.append(0)
    
    rand = random()
    distrib0 = a02/(a02 + a01)
    distrib1 = a01/(a02 + a01)
    total0 = distrib0
    total1 = distrib0 + distrib1
    
    ### goes to the next time (ie. t+2)
    if (rand <= total0):
        predicted_states.append(2)
        
    elif (rand <= total1):
        predicted_states.append(1)

    
    if (predicted_states[-1] == 2):
        predicted_states.append(0)

    elif (predicted_states[-1] == 1):
        predicted_states.append(0)    


if (predicted_states[-1] == 1):
    rand = random()
    
    distrib0 = a10/(a10 + a12)
    distrib1 = a12/(a10 + a12)
    total0 = distrib0
    total1 = distrib0 + distrib1
    
    ### goes to the next time (ie. t+2)
    if (rand <= total0):
        predicted_states.append(0)
    
    elif (rand <= total1):
        predicted_states.append(2)
    

if (predicted_states[-1] == 2):
    predicted_states.append(2)
    
    rand = random()
    #distrib0 = a00
    distrib1 = a20 + a21
    distrib2 = a20 + a21 + a22
    
    ### goes to the next time (ie. t+2)
    if (rand <= distrib1):
        predicted_states.append(1)
    
    elif (rand <= distrib2):
        predicted_states.append(0)

    
    if (predicted_states[-1] == 1):
        predicted_states.append(2)

    elif (predicted_states[-1] == 0):
        predicted_states.append(2)        




############### Time to compute the feature vector at time t
E_k = maxima[-1]   
E_k_1 = maxima[-2]
E_k_2 = maxima[-3]
E_k_3 = maxima[-4]
E_k_4 = maxima[-5]

####### Define 'check new f1'
def check_new_f1(E_k, E_k_1, E_k_2, E_k_3, E_k_4):
    if (E_k > E_k_1):
        return 1
    
    elif (E_k < E_k_1):
        return -1

####### Define 'check new f2'
def check_new_f2(E_k, E_k_1, E_k_2, E_k_3, E_k_4):
    if (E_k_4 < E_k_2 < E_k and E_k_3 < E_k_1):
        return 1
    
    elif (E_k_4 > E_k_2 > E_k and E_k_3 > E_k_1):
        return -1
    
    else: return 0

####### Define 'check new f3'
def check_new_f3(E_k, E_k_1, E_k_2, E_k_3, E_k_4):
    #VWAPkBidSpread = E_k - output_bid_zigzag[-1]
    #VWAPkAskSpread = output_ask_zigzag[-1] - E_k
    
    #phi_new = VWAPkBidSpread - VWAPkAskSpread


check_new_f1(E_k, E_k_1, E_k_2, E_k_3, E_k_4)
check_new_f2(E_k, E_k_1, E_k_2, E_k_3, E_k_4)
check_new_f3(E_k, E_k_1, E_k_2, E_k_3, E_k_4)




###############################################################################
###############################################################################
####################### TO DO AFTER
expected_returns = np.dot(model.transmat_, model.means_)
returns_columnwise = list(zip(*expected_returns))
returns = returns_columnwise[0] 

returns_first_row = returns[0]      # returns if first hidden state
returns_second_row = returns[1]     # returns if second hidden state
returns_third_row = returns[2]      # returns if third hidden state
"""
""" Tricky part in which we have to find the most likely sequence
    of hidden states based on various factors:
        1) the current hidden state 
        2) the transition matrix
        3) the feature vector (ie. zigzags and order book info)
            ---> find the most likely sequence from the past data ??? 
"""
"""
#state = []
predicted_prices = []
lastN = 50

current_price = tradeprice[-1]
state = hidden_states[-lastN]
current_date = 0
predicted_prices.extend((current_date, current_price + returns[state]))

lastN_new = lastN-1

for ite in range(lastN_new):
    state = hidden_states[-lastN_new+ite]
    current_price = predicted_prices[-1]
    current_date = ite+1
    predicted_prices.extend((current_date, current_price + returns[state]))

print(predicted_prices)
"""