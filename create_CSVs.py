import csv


with open('districtNames.csv', 'w') as f_csv:
    writer = csv.writer(f_csv, delimiter='\n')    
    writer.writerow(['districts'])
    f_csv.flush()
    writer.writerow(districts)


with open('districts.csv', 'w') as f_csv:
    for d in districts:
        writer = csv.writer(f_csv, delimiter='\t')    
        writer.writerow(['district','lat','lon'])
        for num in range(len(districts)):
            writer.writerow([districts[num],xx[num],yy[num]])
        f_csv.write('\n')
        f_csv.flush()
            

ds_temporal_noAgg.to_csv('memos_breakdown.csv',sep=',',header=True, index=True)
            
with open('memos_breakdown.csv', 'w') as f_csv:
    for d in districts:
        writer = csv.writer(f_csv, delimiter='\t')    
        writer.writerow(['district','lat','lon'])
        for num in range(len(districts)):
            writer.writerow([districts[num],xx[num],yy[num]])
        f_csv.write('\n')
        f_csv.flush()
 