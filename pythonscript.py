import xlrd

############################################ GET PAIR'S TRADE DATES
workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/bitcoin/data/20160420-Kraken.BTCEUR-Trades.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table1 = []
counter = 1

for x in range(total_rows):
    table1.append(worksheet.cell(x,1).value)

n = len(table1)
s = 'btceur_date.txt'

with open(s,'w') as f:
    while (counter<n):
        line = str(table1[counter]+'\n')
        f.write(line[14]+line[15]+line[17]+line[18]+line[20]+line[21]
                +line[23]+line[24]+line[25]+line[26]+line[27]
                    +line[28]+line[29]+line[30])
        counter += 1

############################################ GET PAIR'S TRADE PRICES
workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/bitcoin/data/20160420-Kraken.BTCEUR-Trades.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table = []
counter = 1

for x in range(total_rows):
    table.append(worksheet.cell(x,4).value)


n = len(table)
s = 'btceur_trade.txt'

with open(s,'w') as f:
    while (counter<n):
        f.write(str(table[counter])+'\n')
        counter += 1
        

############################################ GET PAIR'S TRADE VOLUMES
workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/bitcoin/data/20160420-Kraken.BTCEUR-Trades.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table = []
counter = 1

for x in range(total_rows):
    table.append(worksheet.cell(x,5).value)


n = len(table)
s = 'btceur_volume.txt'

with open(s,'w') as f:
    while (counter<n):
        f.write(str(table[counter])+'\n')
        counter += 1


#################################### GET RID OF THE EMPTY LINES
def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

############################################ GET ORDER BOOK'S TRADE DATES
workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/20160420-Kraken-OrderBooks.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table2 = []
counter = 0

for x in range(total_rows):
    table2.append(worksheet.cell(x,0).value)

n = len(table2)
s = 'btceur_orderbook_date.txt'

with open(s,'w') as f:
    while (counter<n):
        #f.write(str(table2[counter])+'\n')
        line = str(table2[counter]+'\n')
        
        if line.strip():
            f.write(line[14]+line[15]+line[17]+line[18]+line[20]+line[21]+
                    line[23]+line[24]+line[25]+line[26]
                    +line[27]+line[28]+line[29]+line[30])
        counter += 1