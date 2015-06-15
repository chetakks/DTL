import csv
with open('/home/aditya_1t/Downloads/HIGGS.csv', 'rb') as f:
    reader = csv.reader(f)
    #print reader[0]
    for row in reader:
        print row
        print row[0]
        print row[1]
        print row[2]
        print row[3]
        print row[4]