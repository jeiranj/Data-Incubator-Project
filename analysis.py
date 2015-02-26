# Before running this, enter pygmaps folder and run: python setup.py build and python setup.py install
import pygmaps
import webbrowser
import pandas as pd
import numpy as np
import scipy as sp
import shapefile
import dill
import itertools
import matplotlib
import time
import os.path
from datetime import *
from pandas.tools.plotting import autocorrelation_plot
from shapely.geometry import Point, shape, Polygon
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import autocorrelation_plot
from random import sample
from sklearn.decomposition import ProjectedGradientNMF
import matplotlib.cm as cm, matplotlib.font_manager as fm

# force headless backend, or set 'backend' to 'Agg'
# in your ~/.matplotlib/matplotlibrc
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# force non-interactive mode, or set 'interactive' to False
# in your ~/.matplotlib/matplotlibrc
matplotlib.pyplot.ioff()
            
def main():
### load the data and do some minor cleaning:
    data_raw = pd.read_csv('irq.csv',low_memory=False)
    data_raw.date = pd.to_datetime(data_raw.date)
    data_raw['type']=data_raw['type'].str.lower()
    data_raw['category']=data_raw['category'].str.lower()
    typeOfUnit_dict = dict.fromkeys(['CIV', 'Civilian'], 'Civilian')
    typeOfUnit_dict.update(dict.fromkeys(['Coalition Forces', 'CF','Coalition'], 'Coalition Forces'))
    typeOfUnit_dict.update(dict.fromkeys(['Iraqi Security Forces','ISF'], 'Iraqi Security Forces'))
    typeOfUnit_dict.update(dict.fromkeys(['Anti-Iraqi Forces','AIF'], 'Anti-Iraqi Forces'))
    data_raw['typeofunit'].replace(typeOfUnit_dict, inplace=True)
    get_week = lambda x: x.isocalendar()[1]
    data = data_raw[pd.notnull(data_raw.lat)]
    data = data[data.dcolor=="RED"]  #focus on REDs
    years = pd.DatetimeIndex(data['date']).year
    weeks = data['date'].apply(get_week)
    months = pd.DatetimeIndex(data['date']).month
    days = pd.DatetimeIndex(data['date']).day
    groupYMD = data.groupby([years, months, days])
    aggregateYMD = groupYMD.agg({"log": len})
    group_names = list()
    group_names = groupYMD.grouper.result_index.tolist()

    xticklocs = [0,151,335,516,700,881,1065,1246,1430,1612,1796,1946,2129]
    xticklabs = list()
    for tmp in xticklocs:
        xticklabs.append(group_names[tmp])
    ax = aggregateYMD['log'].plot(color='#b60f2e', alpha=0.7,linestyle='None', marker='o')
	start, end = ax.get_xlim()
	ax.set_xticks(xticklocs)
	ax.set_xticklabels(xticklabs)
	plt.xticks(rotation=30)
	plt.xlabel('(year, month, day)', fontsize=30, family='rockwell')
	plt.ylabel('Number of memos', fontsize=30, family='rockwell')
	fig_name = 'EventCount'
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.setp(ax.spines.values(), color='grey')
	plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='lightgrey')
	fig = plt.gcf()
	fig.set_size_inches(18.5,10.5)
	plt.savefig(fig_name)            
    plt.close(fig) 
    
    fig = plt.figure()
	autocorrelation_plot(aggregateYMD['log'],color='#b60f2e',alpha=0.7,linewidth=8.0)
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.xlabel('lag', fontsize=30, family='rockwell')
	plt.ylabel('Autocorrelation', fontsize=30, family='rockwell')
	fig.suptitle('')
	fig_name = 'ACF'
	fig = plt.gcf()
	fig.set_size_inches(18.5,10.5)
	plt.savefig(fig_name)  
	plt.close(fig) 
    
	print 'Forming cities dataframe...'
	cities = pd.DataFrame(index=['Samarra','Baghdad','Ramadi','Kirkuk','Arbil','Mosul','Karbala','Najaf','Nasiriyah','Basrah','Amarah','Fallujah','Tikrit','Kut'],columns=['lat','lon'])
	cities.lat =[34.1983,33.325,33.4167,35.4667,36.1911,36.34,32.616732,32,31.05,30.5,31.8333,33.35,34.6,32.5056]
	cities.lon=[43.8742,44.422,43.3,44.3167,44.0092,43.13,44.0333,44.33,46.2667,47.8167,47.15,43.7833,43.6833,45.8247]
	cities.index.name = 'cities'
	# set location of city names for annotating the maps in the later stages
	annotation_font = fm.FontProperties(family='rockwell', size=18)
	xyt = {}
	for num in range(0,cities.shape[0]):
		xyt[cities.index.values[num]]=(cities.ix[num].lon , cities.ix[num].lat + .1)
	xyt['Ramadi'] = (42.2,33.6167)
	xyt['Fallujah'] = (42.7,33)
	cities.to_csv('cities.csv')
    
# ##### plot monthly google maps of events:
    do_or_die = raw_input('Should we make google maps: (Y or N)')
    if do_or_die == 'Y':
        for date in group_names:
            print "date is ",str(date)
            mymap = produce_gmaps(groupYMD.get_group(date))
            mymap.draw('mymap'+str(date)+'.draw.html') 
            url = 'mymap'+str(date)+'.draw.html'
#     ##### save histogram of happenings:
# #         save_hist_events(data,'')
# #         save_hist_events(data,'RED')
# #         save_hist_events(data,'BLUE')
# #         save_hist_events(data,'GREEN')

    print '======= Reading district locations...'
    sf = shapefile.Reader("IRQ_adm2.shp")
    shapes = sf.shapes()
    districts = list()
    num = 0
    for num in range(0,len(shapes)):
        districts.append(sf.records()[num][6].strip())
    xx = {}  #longitude
    yy = {}  #latitude
    for num in range(0,len(shapes)):
        xx[num],yy[num] = zip(*shapes[num].points)    

	do_or_die = raw_input('Should we read the location of roads in Iraq: (Y or N)')
	if do_or_die == 'Y':
		print '======= Reading road locations...'
		sr = shapefile.Reader("IRQ_roads.shp")
		shapes_roads = sr.shapes()
		roads = list()
		for num in range(0,len(shapes_roads)):
			roads.append(sr.records()[num][1].strip())
		xr = {}  #longitude
		yr = {}  #latitude
		for num in range(0,len(shapes_roads)):
			xr[num],yr[num] = zip(*shapes_roads[num].points)  
		road_dict = {'Primary Route':1, 'Secondary Route':0.5,'Unknown':0}
        
### Daily analysis within districts using shapefile IRQ_adm2
    do_or_die = raw_input('Should we find the points in each district: (Y or N)')
    if do_or_die == 'Y':
    #if not(os.path.isfile('ds_aggregate.pkl')):
        print 'AGGREGATING DATA...'
        points_in_dists = aggregate_by_region(data,sf,'district')
        with open('ds_aggregate.pkl', 'wb') as f:
            dill.dump(points_in_dists, f)
    
    do_or_die = raw_input('Should we aggregate the data district-wise: (Y or N)')
    if do_or_die == 'Y':
        print '=====Opening district data'
        with open('ds_aggregate.pkl', 'rb') as f:
            points_in_dists = dill.load(f)
        print 'TEMPORALLY AGGREGATING DATA BY DISTRICTS...'
        df = pd.DataFrame(np.zeros((len(group_names),len(districts))),columns=districts,index=group_names)
        df.index.name = 'date'
        df.reset_index(inplace = True)
        for num in range(0,len(group_names)):
            group_indices = groupYMD.grouper.indices[group_names[num]]
            print '=========Reading date', group_names[num]
            for g_i in group_indices:
                dist_i = [dist for dist, points in points_in_dists.items() if data.index.values[g_i] in points]
                if dist_i == []:
                    continue
                df.loc[num,dist_i] = int(df.loc[num,dist_i].values)+1
        with open('ds_temporal.pkl', 'wb') as f:
            dill.dump(df, f)

### Daily analysis within districts using shapefile IRQ_adm2, and NOT aggregating events
    do_or_die = raw_input('Should we form the temporal districts data--no aggregating: (Y or N)')
    if do_or_die == 'Y':
        print 'TEMPORALLY ORGANIZING DATA BY DISTRICTS (NOT AGGREGATING)...'
        with open('ds_aggregate.pkl', 'rb') as f:
            points_in_dists = dill.load(f)
        L = len(list(itertools.chain.from_iterable(points_in_dists.values())))
        df = pd.DataFrame(np.zeros((L,len(districts))),columns=districts)
        df.insert(0, 'date', np.zeros(L))
        df['date'] = df['date'].astype(object)
        ind = 0
        for date in group_names:
            group_indices = groupYMD.grouper.indices[date]
            print '=========Reading date', date
            for g_i in group_indices:
                dist_i = [dist for dist, points in points_in_dists.items() if data.index.values[g_i] in points]
                if dist_i == []:
                    continue
                df.loc[ind,dist_i] = 1
                df.loc[ind,'index'] = g_i
                df.loc[ind,'date'] = str(date)
                ind = ind+1
        with open('ds_temporal_noAgg.pkl', 'wb') as f:
            dill.dump(df, f)
    

### Plot for general colorbar (min and max obtained from log of aggregated time series) 
    ds_temporal = pd.read_pickle('ds_temporal.pkl')
    ds_temporal.set_index('date',inplace=True)
    do_or_die = raw_input('Should we make district spatiotemporal maps: (Y or N)')
    if do_or_die == 'Y':
        norm = matplotlib.colors.Normalize(vmin=0,vmax=np.log(max(ds_temporal.apply(max, axis=0))))
        m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
        m._A = []
        plt.colorbar(m)
        plt.ion()
        plt.show()
        for date in ds_temporal.index.values:
            fig = plt.figure(figsize=(13,10))
            norm = matplotlib.colors.Normalize(vmin=0,vmax=np.log(max(ds_temporal.apply(max, axis=0))))
            m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
            m._A = []
            clrbr = plt.colorbar(m)
            clrbr.set_label("log(number of memos)")
            ax = clrbr.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(family='times new roman',size=22)
            text.set_font_properties(font)
            for c in clrbr.ax.get_yticklabels():
                c.set_fontsize(16)
                c.set_family("times new roman")
            for num in range(0,len(sf.shapes())):
                cval = np.log(int(ds_temporal.loc[[date],districts[num]].values))
                plt.fill(xx[num],yy[num],edgecolor='k',color=m.to_rgba(cval))
            plt.title('(year, month, day)=%s'%str(date),fontsize=22, fontname="times new roman")
            plt.axis('off')
            plt.draw()
            figname = 'districtsMap%s'%str(date)
            plt.savefig(figname) 
            plt.close(fig)        
    
    do_or_die = raw_input('Should we make district spatiotemporal maps with points: (Y or N)')
    if do_or_die == 'Y':
        with open('ds_aggregate.pkl', 'rb') as f:
            points_in_dists = dill.load(f)
        plt.ion()
        plt.show()
        for date in group_names:
            fig = plt.figure(figsize=(12,11))
            norm = matplotlib.colors.Normalize(vmin=0,vmax=0)
            m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
            m._A = []
            for num in range(0,len(sf.shapes())):
                #cval = 0
                #plt.fill(xx[num],yy[num],edgecolor='k',color=m.to_rgba(cval))
                plt.fill(xx[num],yy[num],edgecolor='k',color='#363636')
            for num in range(0,len(shapes_roads)):
                plt.plot(xr[num],yr[num],color='gray',linewidth=road_dict[roads[num]])
            plt.title('(year, month, day)=%s'%str(date),fontsize=22, fontname="times new roman")
            plt.axis('off')
            ix_points_in_iraq = [item for sublist in list(points_in_dists.values()) for item in sublist]
            group_indices = groupYMD.grouper.indices[date]
            print '=========Reading date', date
            for g_i in group_indices:
                if g_i in ix_points_in_iraq:
                    plt.plot(data.iloc[g_i].lon,data.iloc[g_i].lat,'o',color='#F47465',alpha = 0.8)
            plt.draw()
            figname = 'districtsPointsMap%s'%str(date)
            plt.savefig(figname) 
            plt.close(fig)  

### PCA on data:
    do_or_die = raw_input('Should we perform PCA: (Y or N)')
    if do_or_die == 'Y':
        print '======= Performing PCA on some X...'
        X = ds_temporal.values.T
        ## fraction of total at each time point:
        X_sum = X.sum(axis=1)
        X = X/X_sum[:,None]
        X[np.isnan(X)] = 0
        ## log(X):
        #X = np.log(X)
        #X[X<-100] = 0.01
        num_sig = 3
        U,D,V = perform_PCA(X,sf,xx,yy,xticklocs,xticklabs,num_sig, num_districts=len(shapes))
        with open('fraction_PCA.pkl', 'wb') as f:
            dill.dump(U, f)
            dill.dump(V, f)
            dill.dump(D, f)
        with open('fraction_PCA.pkl', 'rb') as f:
            U = dill.load(f)
            V = dill.load(f)
            D = dill.load(f)
    ### Plot temporally-projected data
        vTx = np.zeros((num_sig,X.shape[0]))
        for num in range(num_sig):
            vTx[num,] = V[num,:].dot(X.T)
        plt.ion()
        plt.show()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(vTx[0,:],vTx[1,:],vTx[2,:], marker='o', s=40, color='blue', alpha=0.4)
        for ii in xrange(0,360,30):
            ax.view_init(elev=10., azim=ii)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.draw()
            figname = 'TemporalProjection_fraction_%s'%str(ii)
            plt.savefig(figname)         
        plt.close(fig)  
    ### Plot spatially-projected data
        uTx = np.zeros((num_sig,X.shape[1]))
        for num in range(num_sig):
            uTx[num,] = U[:,num].dot(X)
        plt.ion()
        plt.show()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(uTx[0,:],uTx[1,:],uTx[2,:], marker='o', s=40, color='blue', alpha=0.4)
        for ii in xrange(0,360,30):
            ax.view_init(elev=10., azim=ii)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.draw()
            figname = 'SpatialProjection_fraction_%s'%str(ii)
            plt.savefig(figname)         
        plt.close(fig)

### NMF on data:
    do_or_die = raw_input('Should we perform NMF: (Y or N)')
    if do_or_die == 'Y':
        print '======= Performing NMF on some X...'
		X = ds_temporal.values.T
		sig=3
		model = ProjectedGradientNMF(n_components=sig, init='random')
		spatial_components = model.fit_transform(X)
		temporal_components = model.components_.T
		# plot and save temporal component figures
		fig, ax = plt.subplots()
		for pc in range(sig):
			plt.plot(temporal_components[:,pc],color='#b60f2e',alpha=0.7,linestyle='None', marker='o')
			plt.ylabel('Number of memos',fontsize=30, family='rockwell')
			ax.set_xticks(xticklocs)
			ax.set_xticklabels(xticklabs,fontsize=18)
			plt.xlim(xticklocs[0], xticklocs[-1])
			plt.xticks(rotation=30)
			plt.xlabel('(year, month, day)', fontsize=30, family='rockwell')
			plt.title('temporal part '+str(pc+1),fontsize=40, family='rockwell')
			plt.tick_params(axis='both', which='major', labelsize=20)
			plt.setp(ax.spines.values(), color='grey')
			plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='grey')
			fig.set_size_inches(18.5,10.5)
			figname = 'NMF_temporals'+str(pc)
			plt.savefig(figname)
			plt.close(fig) 
		# plot and save spatial component figures
		norm = matplotlib.colors.Normalize(vmin=0,vmax=10)
		m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
		m._A = []
		fig = plt.figure(figsize=(14,11.5))
		plt.colorbar(m)
		plt.ion()
		for pc in range(sig):
			fig = plt.figure(figsize=(14,11.5))
			ax = fig.add_subplot(111)
			norm = matplotlib.colors.Normalize(vmin=spatial_component[:,pc].min(),vmax=(spatial_component[:,pc]/20).max())
			m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
			m._A = []
			clrbr = plt.colorbar(m)  
			ax = clrbr.ax
			text = ax.yaxis.label
			font = matplotlib.font_manager.FontProperties(family='rockwell',size=20)
			text.set_font_properties(font)
			for c in clrbr.ax.get_yticklabels():
				c.set_fontsize(20)
				c.set_family("rockwell")
			for num in range(0,len(sf.shapes())):
				cval = spatial_component[num,pc]
				plt.fill(xx[num],yy[num],edgecolor='grey',color=m.to_rgba(cval))
			for num in range(0,cities.shape[0]):
				plt.plot(cities.ix[num].lon,cities.ix[num].lat,'o',color='k',markersize=10)
				plt.annotate(cities.index.values[num], xy=(cities.ix[num].lon, cities.ix[num].lat), xytext=xyt[cities.index.values[num]],fontproperties=annotation_font,xycoords='data')
			plt.title('Spatial part '+str(pc+1),fontsize=30, fontname="rockwell")
			plt.axis('off')
			plt.draw()
			figname = 'NMF_spatial_factor%s'%str(pc)
			plt.savefig(figname)        
			plt.close(fig)



# #### Temporal smoothing
    do_or_die = raw_input('Should we perform l-2 Temporal smoothing: (Y or N)')
    if do_or_die == 'Y':
        print '======= Performing l2 temporal smoothing...'
        ## Prepare the data:
        # this dataframe contains the number of events in each district aggregated per day.
        ds_temporal = pd.read_pickle('ds_temporal.pkl')
		ds_temporal.set_index('date',inplace=True)
		# this dataframe contains either 0s or 1s where 1 corresponds to an event, 0 to no event. (the same as the above dataframe, but un-aggregated).
		ds_temporal_noAgg = pd.read_pickle('ds_temporal_noAgg.pkl')
		ds_temporal_noAgg.set_index('index',inplace=True)
		dates = [datetime.strptime(i,'(%Y, %m, %d)') for i in ds_temporal_noAgg['date'].values]
		ds_temporal_noAgg.drop('date', axis=1, inplace=True)
		ds_temporal_noAgg.insert(0, 'date', dates)
		group_data , aggregate_data = groupAndAggregate(ds_temporal_noAgg)
		groupNames_data = list()
		groupNames_data = [name for name, group in group_data]         
        # Produce csv files of time series of event counts for each district - to be used for d3 visualization
        csvs = raw_input('Should we produce district .CSVs: (Y or N)')
        if csvs == 'Y':
            print '======= Writing CSV files for each district...'
            for district in districts[17:]:
                district_data = form_var_stabilized_district_df(district,aggregate_data)
                tmp = [list(data.index.values[group_data.get_group(groupNames_data[i])[district].loc[(group_data.get_group(groupNames_data[i])[district] > 0).values].index.values.astype(int)]) for i in range(len(groupNames_data))]
                district_data['lat'] = [list(data.loc[i].lat.values) for i in tmp]
                district_data['lon'] = [list(data.loc[i].lon.values) for i in tmp]
                district_data['type'] = [list(data.loc[i].type.values) for i in tmp]
                district_data['category'] = [list(data.loc[i].category.values) for i in tmp]
                district_data.to_csv('memos_aggregate_'+district+'.csv',tupleize_cols=True,sep=',',header=True,index=True,index_label=('year','month','day'))
                print 'Wrote the csv file for', district
        # divide to two subsets: learn for training and testing and the second subset is to validation
        T = len(ds_temporal_noAgg.index)
		percent_learning = 0.1
		row_learn = sorted(sample(ds_temporal_noAgg.index, int(np.floor(percent_learning*T))))
		ds_learn = ds_temporal_noAgg.loc[row_learn]
		ds_validate = ds_temporal_noAgg.drop(row_learn)
		group_learn, aggregate_learn = groupAndAggregate(ds_learn)
		group_validate, aggregate_validate = groupAndAggregate(ds_validate)
		groupNames_learn = [name for name, group in group_learn]
		groupNames_validate = [name for name, group in group_validate]
		# choose the district for kernel smoothing:
		district = 'Adhamiya'
		district_learn = form_var_stabilized_district_df(district,aggregate_learn)
		district_validate = form_var_stabilized_district_df(district,aggregate_validate)
   		index_learn = convert_to_t(district_learn)
		index_validate = convert_to_t(district_validate)
		data_learn = pd.DataFrame(district_learn.values,index = index_learn, columns=['X','N','P','Z'])
		data_validate = pd.DataFrame(district_validate.values,index = index_validate, columns=['X','N','P','Z'])
		data_validate.to_csv('validate.csv',sep=',',header=False, index=True)
		data_learn.to_csv('learn.csv',sep=',',header=False, index=True)
  	    ## First, perform simple l2-smoothing:
  	    x = data_validate.index.values
  	    y = data_validate.Z.values
  	    y_hat,y_var,sig2,l = l2_smooth(x, y, 3)
		yhat_plus_std = yhat + 1.96
		yhat_minus_std = 
		# see that hyper parameters have converged:
		fig, axarr = plt.subplots(1,2)
		axarr[0].plot(l)
		axarr[0].set_title('$\lambda$')
		axarr[1].plot(sig2)
		axarr[1].set_title('$\sigma^2$')
		fig.set_size_inches(15,6)
		# check how you smoothed the time series:
		ymean_minus_std = y_mean-1.96*(np.power(y_var,0.5))
		ymean_plus_std = y_mean+1.96*(np.power(y_var,0.5))
		plt.plot(x,y,'.')
		plt.plot(x,y_mean,color='red')
		plt.fill_between(x,ymean_plus_std,ymean_minus_std,alpha=0.5)
		xtix = [0,182,366,547,731,912,1096,1277,1461,1643,1827,2008,2191]
		xtixlabs = ['(2004, 1, 1)','(2004, 7, 1)','(2005, 1, 1)','(2005, 7, 1)','(2006, 1, 1)','(2006, 7, 1)','(2007, 1, 1)','(2007, 7, 1)','(2008, 1, 1)','(2008, 7, 1)','(2009, 1, 1)','(2009, 7, 1)','(2009, 12, 31)']
		plt.xticks(xtix,xtixlabs)
		plt.xticks(rotation=30)
		plt.xlim(xticklocs[0], xticklocs[-1])
		plt.show()

#     do_or_die = raw_input('Should we perform l-2 Temporal smoothing: (Y or N)')
#     if do_or_die == 'Y':
# 		number_of_gibbs_samples = 1000
# 		number_of_burned_samples = 200
# SMG code goes here      
        

### Adjacency matrix
def get_adjacency_matrix(shapes):
    adj_matrix = np.eye(len(shapes))
    for num1 in range(0,len(shapes)):
        print num1
        for num2 in range(num1+1,len(shapes)):
            for num1_i in range(0,len(shapes[num1].points)):
                if shapes[num1].points[num1_i] in shapes[num2].points:
                    adj_matrix[num1,num2] = 1
                    adj_matrix[num2,num1] = 1
                    continue
      

def produce_gmaps(data):
    mymap = pygmaps.maps(33.5, 43.5, 6.5)
    dcolor_dict = {'RED':'#b60f2e', 'BLUE':'#4EA8EC','GREEN':'#78B678'}
    for ind in data.index.values:
        lat = data.loc[ind].lat
        lon = data.loc[ind].lon
        color = dcolor_dict[data.loc[ind].dcolor]
        mymap.addradpoint(lat,lon, 4000, color)
    return mymap 
    

def save_hist_events(data,filename):
    if filename in ('RED','BLUE','GREEN'):
        data = data[data.dcolor==filename]
        col = filename.lower()
    elif filename == '':
        col = 'gray'
    years = pd.DatetimeIndex(data['date']).year
    months = pd.DatetimeIndex(data['date']).month
    data.groupby([years,months])
    groupYearMonth = data.groupby([years,months])
    aggregateYearMonth = groupYearMonth.agg({"log": len})    
    
    aggregateYearMonth['log'].plot(kind="bar",color=col)
    plt.xticks(rotation=50)
    plt.xlabel('(year, month)', fontsize=40)
    plt.ylabel('Frequency', fontsize=40)
    plt.title('Number of events', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=10)
    fig1 = plt.gcf()
    fig1.set_size_inches(18.5,10.5)
    hist_name = 'eventCount%s'%filename
    plt.savefig(hist_name)         
    plt.close(fig1)
    

def aggregate_by_region(data,sf,type='district'):  #or 'governorate'
    shapes = sf.shapes()
    regions = list()
    for num in range(0,len(shapes)):
        if type == 'district':
            regions.append(sf.records()[num][6].strip())
        elif type == 'governorate':
            regions.append(sf.records()[num][4].strip())
    xx = {}  #longitude
    yy = {}  #latitude
    for num in range(0,len(shapes)):
        xx[num],yy[num] = zip(*shapes[num].points)
    points_in_regions = dict.fromkeys(regions)
    for num in range(0,len(shapes)):
        poly = Polygon(zip(xx[num],yy[num]))
        points_in_regions[regions[num]] = []
        print '=========Reading region',num
        for num_point in data.index.values:
            point = Point(data.loc[num_point].lon,data.loc[num_point].lat)
            if point.within(poly) == True:
                points_in_regions[regions[num]].append(num_point)
    return points_in_regions

 
 ### Produce plots and maps for certain key and value in dataframe column
def zoom_on_column(data,column_name,column_value):
    data = data[data[column_name]==column_value]
    years = pd.DatetimeIndex(data['date']).year
    months = pd.DatetimeIndex(data['date']).month
    days = pd.DatetimeIndex(data['date']).day
    groupYMD = data.groupby([years, months, days])
    aggregateYMD = groupYMD.agg({"log": len})
    group_names = list()
    group_names = groupYMD.grouper.result_index.tolist()
    ax = aggregateYMD['log'].plot(color='red')
    start, end = ax.get_xlim()
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xticklabs)
    plt.xticks(rotation=50)
    plt.xlabel('(year, month, day)', fontsize=40, family='times new roman')
    plt.ylabel('Number of ' +column_value+ ' memos', fontsize=40, family='times new roman')
    figname = column_value+'Count'
    plt.tick_params( labelsize=14)
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    plt.savefig(figname)            
 
 
### PCA on logarithm of RED data           
def perform_PCA(X,sf,xx,yy,xticklocs,xticklabs,num_sig=3, num_districts=102):
    # if X.shape[0] != num_districts:
#         X = X.T
    T = X.shape[1]
    N = X.shape[0]
    meanX = np.mean(X,axis=1)
    meanX = meanX.reshape(len(meanX),1)
    demeaned = (X-meanX)
    U, d, V = np.linalg.svd(demeaned/(T**0.5))#,full_matrices = True)
    D = np.zeros((num_sig,num_sig), dtype=float)
    D[:len(d[0:num_sig]), :len(d[0:num_sig])] = np.diag(d[0:num_sig,])
    reduced_X = (meanX + ((U[:,0:num_sig].dot(D)).dot(V[0:num_sig,:]))*(T**0.5))
### Plot eigenvalues
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(121)
    ax.hist(d**2,bins=100)
    ax.set_ylabel('Count',fontsize=18)
    ax.set_xlabel('Eigenvalue',fontsize=18)
    ax = fig.add_subplot(122)
    ax.plot(d**2)
    ax.set_ylabel('Eigenvalue',fontsize=18)
    figname = 'Eigenvalues'
    plt.savefig(figname) 
    plt.close(fig)    
### Plot data and transformed data
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    ax.plot(X.T)
    ax.set_ylabel('Number of memos',fontsize=18)
    ax.set_xlabel('Time',fontsize=18)
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xticklabs,fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(rotation=50,fontname='times new roman')
    plt.xlabel('(year, month, day)', fontsize=20, family='times new roman')
    plt.ylabel('log(number of memos) before PCA', fontsize=20, family='times new roman')
    plt.xlim(xticklocs[0], xticklocs[-1])
    ax = fig.add_subplot(122)
    ax.plot(reduced_X.T)
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xticklabs,fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(rotation=50,fontname='times new roman')
    plt.xlabel('(year, month, day)', fontsize=20, family='times new roman')
    plt.ylabel('log(number of memos) after PCA', fontsize=20, family='times new roman')
    plt.xlim(xticklocs[0], xticklocs[-1])
    figname = 'PCA_log_data'
    plt.savefig(figname) 
    plt.close(fig)
### Plot temporal PCs
    fig = plt.figure(figsize=(20, 10))
    plt.plot(V[0,:])
    plt.plot(V[1,:],color='red')
    plt.plot(V[2,:],color='green')
    plt.ylabel('V (temporal PC)')
    plt.legend(['first PC', 'second PC', 'third PC'])
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xticklabs,fontsize=18)
    plt.xlim(xticklocs[0], xticklocs[-1])
    plt.xticks(rotation=50,fontname='times new roman')
    plt.xlabel('(year, month, day)', fontsize=20, family='times new roman')
    plt.ylabel('Temporal Pcs', fontsize=20, family='times new roman')
    figname = 'PCA_temporal_PCs'
    plt.savefig(figname) 
    plt.close(fig)
### Plot spatial PCs
    norm = matplotlib.colors.Normalize(vmin=-10,vmax=10)
    m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
    m._A = []
    plt.colorbar(m)
    plt.ion()
    plt.show()
    for pc in range(num_sig):
        fig = plt.figure(figsize=(13,10))
        norm = matplotlib.colors.Normalize(vmin=U[:,pc].min(),vmax=U[:,pc].max())
        m = matplotlib.cm.ScalarMappable(norm,cmap='Reds')
        m._A = []
        clrbr = plt.colorbar(m)
        ax = clrbr.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(family='times new roman',size=20)
        text.set_font_properties(font)
        for c in clrbr.ax.get_yticklabels():
            c.set_fontsize(16)
            c.set_family("times new roman")
        for num in range(0,len(sf.shapes())):
            cval = U[num,pc]
            plt.fill(xx[num],yy[num],edgecolor='k',color=m.to_rgba(cval))
        plt.title('Spatial PC=%s'%str(pc),fontsize=20, fontname="Times New Roman")
        plt.axis('off')
        plt.draw()
        figname = 'PCA_spatial_PC%s'%str(pc)
        plt.savefig(figname)        
        plt.close(fig)
    Dfull = np.zeros((U.shape[1],V.shape[0]), dtype=float)
    Dfull[:len(d), :len(d)] = np.diag(d)
    return U, Dfull, V
 
### variance stabilization transform for fraction of events in a given district        
def form_var_stabilized_district_df(district,aggregate_data):
    df = aggregate_data[district].to_frame(name=district)
    df['All'] = aggregate_data.sum(axis=1)
    df['Fraction'] = (df[district]+2)/(df['All']+4)
    df['NormalTrans'] = 2*np.arcsin(((df[district]+0.25)/(df['All']+0.5))**0.5)
    return df
    
### group and aggregate data by year, month and day
def groupAndAggregate(data):
    yr = pd.DatetimeIndex(data['date']).year
    mn = pd.DatetimeIndex(data['date']).month
    dy = pd.DatetimeIndex(data['date']).day
    grouped = data.groupby([yr, mn, dy])
    aggregated = grouped.agg(np.sum)
    return grouped, aggregated
    
### Convert time to temporal index for a given date wrt 1/1/2004 which is the beginning of the data
def get_t_for_one(data,index1):
    t = (pd.datetime(data.index.values[index1][0],data.index.values[index1][1],data.index.values[index1][2])-pd.datetime(2004,1,1)).days
    return t

### Get temporal indices for all data points    
def convert_to_t(data):
    delta = np.asarray([get_t_for_one(data, index1) for index1 in range(len(data.index.values))])
    return delta

### l2 smoothing with Gibbs sampler
def l2_smooth(t, y, order):
    '''
    t and y are N-dimensional vectors.
    $\mathcal{l}_2$ smoothing is a convex problem with a well-formulated solution: given data vector $y \in \mathbb{R}^N$, $\hat{y} = (I + l D^T D)^{-1} y$. 
    D is the sequential difference-operator matrix.
    Regularization parameter (l) controls the degree of smoothness. 
    This prior consists of smoothness penalty of degree=order which is the order of derivative of the time series. In lasso, order=3 is equivalent to to a pieciwise quadratic time series.
    Appropriate conjugate hyperpriors are adopted for variance and regularization parameter.
    Gibbs sampler is used to sample from the posterior.
    '''
    N = len(t)
    I = sp.sparse.spdiags(np.ones(N),0,N,N)
    D = (sp.sparse.spdiags(np.ones(N), 0, N, N)-sp.sparse.spdiags(np.ones(N), 1, N, N))
    count = 1
    Dk = D
    while count<order:
        Dk = Dk*D
        count = count+1
    Dk = Dk[:-order,:]
    sig2 = [.5]
    k = .01
    e = .01
    l = [500]
    r = .2
    d = .1
    x_mean = np.zeros(N)
    x_var = np.zeros(N)
    burn_samples = 200
    for num in range(800):
        if num%100==0:
            print "Gibbs sampler is sampling; sample number is ", num
        # 1. sample x from normal posterior distribution with mean and covariance; p(x|y,sigma^2,lambda)
        spm = sp.sparse.csc_matrix(I+l[num]*Dk.transpose()*Dk)
        lu = sp.sparse.linalg.splu(spm)
        inv_spm = (lu.solve(np.eye(N))).T
        Sigma_hat = sig2[num]*inv_spm
        mu_hat = np.dot(inv_spm, y)
        C = sp.linalg.cholesky(Sigma_hat)
        x_ = np.dot(C,np.random.normal(0,1,N))+mu_hat
        x_ = np.squeeze(np.asarray(x_))
        j = (num-burn_samples)
        if j>0:
            x_mean=(j-1)*x_mean/j+x_/j;
            x_var=(j-1)*x_var/j+np.power((x_-x_mean),2)/j;
        # 2. sample sigma^2 from inv-gamma posterior distribution with parameters k,e; p(sigma^2|y,x,lambda) where prior p(sigma^2) is an inv-gamma.
        # invgamma.pdf(x;k,e) = x**(-k-1) / gamma(k) * (e**k) * exp(-e/x)
        # if x~invgamma(k,1) then e*x~invgamma(k,e)
        k_ = k+N-order/2.
        e_ = e+(l[num]*(np.linalg.norm(Dk*x_,2)**2) + np.linalg.norm(y-x_,2)**2)/2.
        sig2.append(e_*sp.stats.invgamma.rvs(k_))
        # 3. sample lambda from gamma posterior distribution with parameters r,d; p(lambda|y,x,sigma^2) where prior p(lambda) is a gamma.
        # gamma.pdf(w;r,d) = (x**(r-1)) / gamma(r) * (d**r) * exp(-d*x)
        r_ = r + (N-order)/2.
        d_ = d + ((np.linalg.norm(Dk*x_,2))**2)/(2*sig2[num])
        l.append(np.random.gamma(r_,1./d_))
    return x_mean, x_var, sig2, l
		


   
     
if __name__ == '__main__':
    main()
