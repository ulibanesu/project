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