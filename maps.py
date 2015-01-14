# Before running this, enter pygmaps folder and run: python setup.py build and python setup.py install
import pygmaps
import webbrowser
import pandas as pd
mymap = pygmaps.maps(33.5, 43.5, 6.5)
#mymap.setgrids(37.42, 37.43, 0.001, -122.15, -122.14, 0.001)
#mymap.addpoint(30.9046960743,47.4441623015, "#0000FF","test") 

data = pd.read_csv('irq.csv')
#data['number'] = range(1, len(data) + 1)
data.date = pd.to_datetime(data.date)
dcolor_dict = {'RED':'#F47465', 'BLUE':'#4EA8EC','GREEN':'#78B678'}

data_goodLoc = data[pd.notnull(data.lat)]
years = pd.DatetimeIndex(data_goodLoc['date']).year
months = pd.DatetimeIndex(data_goodLoc['date']).month
groupYearMonth = data_goodLoc.groupby([years,months])
aggregateYearMonth = groupYearMonth.agg({"log": len})

for year in years:
    print "year is ",str(year)
    for month in months:
        for ind in groupYearMonth.get_group((year,month)).index.values:
            lat = data.ix[ind].lat
            lon = data.ix[ind].lon
            color = dcolor_dict[data.ix[ind].dcolor]
            mymap.addradpoint(lat,lon, 1000, color)
        mymap.draw('mymap'+str(year)+'-'+str(month)+'.draw.html') 
        url = 'mymap'+str(year)+'-'+str(month)+'.draw.html'
webbrowser.open_new_tab(url) 

# year=2004
# month=1
# for ind in groupYearMonth.get_group((year,month)).index.values:
#     lat = data.ix[ind].lat
#     lon = data.ix[ind].lon
#     color = dcolor_dict[data.ix[ind].dcolor]
#     mymap.addradpoint(lat,lon, 2000, color)
# mymap.draw('mymap'+str(year)+'-'+str(month)+'.draw.html') 
# url = 'mymap'+str(year)+'-'+str(month)+'.draw.html'
# webbrowser.open_new_tab(url) 
