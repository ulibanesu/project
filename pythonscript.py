import xlrd

############################################ GET PAIR'S DATES
workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/bitcoin/data/20160420-Kraken.BTCEUR-Trades.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table1 = []
counter = 1

for x in range(total_rows):
    table1.append(worksheet.cell(x,3).value)

n = len(table1)
s = 'btceur_date.txt'

with open(s,'w') as f:
    while (counter<n):
        f.write(str(counter)+'\n')
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