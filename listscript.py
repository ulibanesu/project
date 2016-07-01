import xlrd

workbook = xlrd.open_workbook('/Users/KHATIBUSINESS/20160420-Kraken-OrderBooks.xlsx')
worksheet = workbook.sheet_by_name('Sheet1')

total_rows = worksheet.nrows
total_cols = worksheet.ncols

table = []
i=0
counter = 0

for x in range(total_rows):
    table.append(worksheet.cell(x,2).value)
    

n = len(table)
while (counter<n):
    print(counter)
    j=0
    k=str(i)
    s = 'antoine' + k + '.json'
    with open(s,'w') as f:
        f.write('[')
        while ((counter + j)<n) and len(table[counter + j])>0:
            f.write(table[counter+j])
            if ((counter+j)<n-1) and (len(table[counter+j+1])>0):
                f.write(', ')
            j+=1
        f.write(']')
    counter = counter + j + 1
    i+=1