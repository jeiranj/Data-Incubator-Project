# Before running this, enter pygmaps folder and run: python setup.py build and python setup.py install
import pygmaps
import webbrowser
import pandas as pd
import numpy as np
import shapefile
import dill
import itertools
import matplotlib
import time
from datetime import *
from shapely.geometry import Point, shape, Polygon
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import autocorrelation_plot
import os.path
from random import sample
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
    ax = aggregateYMD['log'].plot(color='red')
    start, end = ax.get_xlim()
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(xticklabs)
    plt.xticks(rotation=50)
    plt.xlabel('(year, month, day)', fontsize=40, family='times new roman')
    plt.ylabel('Number of memos', fontsize=40, family='times new roman')
    plt.tick_params(labelsize=14)
    plt.xlim(xticklocs[0], xticklocs[-1])
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    figname = 'eventCount'
    plt.savefig(figname) 
    plt.close(fig) 
    
# ##### plot monthly google maps of events:
    do_or_die = raw_input('Should we make google maps: (Y or N)')
    if do_or_die == 'Y':
        for date in group_names:
            print "date is ",str(date)
            mymap = produce_gmaps(groupYMD.get_group(date))
            mymap.draw('mymap'+str(date)+'.draw.html') 
            url = 'mymap'+str(date)+'.draw.html'
        #webbrowser.open_new_tab(url) 
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

# print '======= Reading road locations...'
#     sr = shapefile.Reader("IRQ_roads.shp")
#     shapes_roads = sr.shapes()
#     roads = list()
#     for num in range(0,len(shapes_roads)):
#         roads.append(sr.records()[num][1].strip())
#     xr = {}  #longitude
#     yr = {}  #latitude
#     for num in range(0,len(shapes_roads)):
#         xr[num],yr[num] = zip(*shapes_roads[num].points)  
#     road_dict = {'Primary Route':1, 'Secondary Route':0.5,'Unknown':0}
#         

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
        #df.insert(0, 'index', np.zeros(L))
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


# #### Temporal smoothing
    do_or_die = raw_input('Should we perform Temporal smoothing: (Y or N)')
    if do_or_die == 'Y':
        print '======= Performing temporal smoothing...'
        ds_temporal = pd.read_pickle('ds_temporal.pkl')
        ds_temporal.set_index('date',inplace=True)
        ds_temporal_noAgg = pd.read_pickle('ds_temporal_noAgg.pkl')
        ds_temporal_noAgg.set_index('index',inplace=True)
        dates = [datetime.strptime(i,'(%Y, %m, %d)') for i in ds_temporal_noAgg['date'].values]
        ds_temporal_noAgg.drop('date', axis=1, inplace=True)
        ds_temporal_noAgg.insert(0, 'date', dates)
        group_data , aggregate_data = groupAndAggregate(ds_temporal_noAgg)
        groupNames_data = list()
        groupNames_data = [name for name, group in group_data]

        csvs = raw_input('Should we produce district .CSVs: (Y or N)')
        if csvs == 'Y':
            print '======= Writing CSV files for each district...'
            for district in districts[17:]:
                district_data = form_distric_df(district,aggregate_data)
                tmp = [list(data.index.values[group_data.get_group(groupNames_data[i])[district].loc[(group_data.get_group(groupNames_data[i])[district] > 0).values].index.values.astype(int)]) for i in range(len(groupNames_data))]
                district_data['lat'] = [list(data.loc[i].lat.values) for i in tmp]
                district_data['lon'] = [list(data.loc[i].lon.values) for i in tmp]
                district_data['type'] = [list(data.loc[i].type.values) for i in tmp]
                district_data['category'] = [list(data.loc[i].category.values) for i in tmp]
                district_data.to_csv('memos_aggregate_'+district+'.csv',tupleize_cols=True,sep=',',header=True,index=True,index_label=('year','month','day'))
                print 'wrote it for', district
        
        # divide to two subsets: learn for training and testing and the second subset is to validate the learning process
        T = len(ds_temporal_noAgg.index)
        percent_learning = 0.1
        row_learn = sorted(sample(ds_temporal_noAgg.index, int(np.floor(percent_learning*T))))
        ds_learn = ds_temporal_noAgg.loc[row_learn]
        ds_validate = ds_temporal_noAgg.drop(row_learn)
        group_learn, aggregate_learn = groupAndAggregate(ds_learn)
        group_validate, aggregate_validate = groupAndAggregate(ds_validate)
        aggregate_learn['All'] = aggregate_learn.sum(axis=1)
        aggregate_validate['All'] = aggregate_validate.sum(axis=1)
        
        # Normally-weighted moving average
        print '=====Performing gaussian kernel smoothing on transformed data....'
        y = district_learn['NormalTrans'].values
        x = [datetime.strptime(str(i),'(%Y, %m, %d)') for i in district_learn['NormalTrans'].index.values]


#         fig, axes = plt.subplots(nrows=4, ncols=1)
#         district_learn['Adhamiya'].plot(ax=axes[0]); axes[0].set_title('Baghdad / LEARN');axes[0].set_xticks(xticklocs); axes[0].set_xticklabels(xticklabs);plt.xticks(rotation=50);
#         district_validate['Adhamiya'].plot(ax=axes[1]); axes[1].set_title('Baghdad / VALIDATE');axes[1].set_xticklabels(xticklabs);axes[1].set_xticks(xticklocs);
#         district_learn['All'].plot(ax=axes[2]); axes[2].set_title('All / LEARN');axes[2].set_xticks(xticklocs); axes[2].set_xticklabels(xticklabs);plt.xticks(rotation=50);
#         district_validate['All'].plot(ax=axes[3]); axes[3].set_title('All / VALIDATE');axes[3].set_xticklabels(xticklabs);axes[3].set_xticks(xticklocs);
#         fig.set_size_inches(10.5,10.5)
#         fig, axes = plt.subplots(nrows=4, ncols=1)
#         district_learn['Fraction'].plot(ax=axes[0]); axes[0].set_title('Baghdad Fraction / LEARN');axes[0].set_xticklabels(xticklabs);axes[0].set_xticks(xticklocs);
#         district_learn['NormalTrans'].plot(ax=axes[1]); axes[1].set_title('Baghdad Transformed / LEARN');axes[1].set_xticklabels(xticklabs);axes[1].set_xticks(xticklocs);
#         district_validate['Fraction'].plot(ax=axes[2]); axes[2].set_title('Baghdad Fraction / VALIDATE');axes[2].set_xticklabels(xticklabs);axes[2].set_xticks(xticklocs);
#         district_validate['NormalTrans'].plot(ax=axes[3]); axes[3].set_title('Baghdad Transformed / VALIDATE');axes[3].set_xticklabels(xticklabs);axes[3].set_xticks(xticklocs);
#         fig.set_size_inches(10.5,15.5)



#         district = 'Adhamiya'
#         district_learn = form_distric_df(district,aggregate_learn)
#         district_validate = form_distric_df(district,aggregate_validate)
#         groupNames_learn = [name for name, group in group_learn]
#         groupNames_validate = [name for name, group in group_validate]

           

# percent_learning = 0.1
# row_learn = sorted(sample(ds_temporal_noAgg.index, int(np.floor(percent_learning*T))))
# ds_learn = ds_temporal_noAgg.loc[row_learn]
# ds_validate = ds_temporal_noAgg.drop(row_learn)
# group_learn, aggregate_learn = groupAndAggregate(ds_learn)
# group_validate, aggregate_validate = groupAndAggregate(ds_validate)
# aggregate_learn['All'] = aggregate_learn.sum(axis=1)
# aggregate_validate['All'] = aggregate_validate.sum(axis=1)
# baghdad_learn = aggregate_learn['Adhamiya'].to_frame(name='Adhamiya')
# baghdad_validate = aggregate_validate['Adhamiya'].to_frame(name='Adhamiya')
# baghdad_learn['All'] = aggregate_learn.sum(axis=1)
# baghdad_learn['Fraction'] = baghdad_learn['Adhamiya']/baghdad_learn['All']
# baghdad_learn['NormalTrans'] = 2*np.arcsin(((baghdad_learn['Adhamiya']+0.25)/(baghdad_learn['All']+0.5))**0.5)
# baghdad_validate['All'] = aggregate_validate.sum(axis=1)
# baghdad_validate['Fraction'] = baghdad_validate['Adhamiya']/baghdad_validate['All']
# baghdad_validate['NormalTrans'] = 2*np.arcsin(((baghdad_validate['Adhamiya']+0.25)/(baghdad_validate['All']+0.5))**0.5)
# groupNames_learn = [name for name, group in group_learn]
# groupNames_validate = [name for name, group in group_validate]


# fig, axes = plt.subplots(nrows=4, ncols=1)
# district_learn['Adhamiya'].plot(ax=axes[0]); axes[0].set_title('Baghdad / LEARN');axes[0].set_xticks(xticklocs); axes[0].set_xticklabels(xticklabs);plt.xticks(rotation=50);
# district_validate['Adhamiya'].plot(ax=axes[1]); axes[1].set_title('Baghdad / VALIDATE');axes[1].set_xticklabels(xticklabs);axes[1].set_xticks(xticklocs);
# district_learn['All'].plot(ax=axes[2]); axes[2].set_title('All / LEARN');axes[2].set_xticks(xticklocs); axes[2].set_xticklabels(xticklabs);plt.xticks(rotation=50);
# district_validate['All'].plot(ax=axes[3]); axes[3].set_title('All / VALIDATE');axes[3].set_xticklabels(xticklabs);axes[3].set_xticks(xticklocs);
# fig.set_size_inches(10.5,10.5)
# 
# fig, axes = plt.subplots(nrows=4, ncols=1)
# district_learn['Fraction'].plot(ax=axes[0]); axes[0].set_title('Baghdad Fraction / LEARN');axes[0].set_xticklabels(xticklabs);axes[0].set_xticks(xticklocs);
# district_learn['NormalTrans'].plot(ax=axes[1]); axes[1].set_title('Baghdad Transformed / LEARN');axes[1].set_xticklabels(xticklabs);axes[1].set_xticks(xticklocs);
# district_validate['Fraction'].plot(ax=axes[2]); axes[2].set_title('Baghdad Fraction / VALIDATE');axes[2].set_xticklabels(xticklabs);axes[2].set_xticks(xticklocs);
# district_validate['NormalTrans'].plot(ax=axes[3]); axes[3].set_title('Baghdad Transformed / VALIDATE');axes[3].set_xticklabels(xticklabs);axes[3].set_xticks(xticklocs);
# fig.set_size_inches(10.5,10.5)
# 
# ax = aggregate_data['Adhamiya'].plot()
# ax.set_xticks(xticklocs)
# ax.set_xticklabels(xticklabs,fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=10)
# plt.xticks(rotation=50)
# plt.xlabel('(year, month, day)', fontsize=20, family='times new roman')
# plt.ylabel('Number of memos', fontsize=20, family='times new roman')
# fig = plt.gcf()
# fig.set_size_inches(18.5,10.5)
    

#     
# T = ds_temporal_noAgg.shape[0]
# percent_learning = 0.2
# row_learn = sorted(sample(ds_temporal_noAgg.index, int(np.floor(percent_learning*T))))
# ds_learn = ds_temporal_noAgg.loc[row_learn].groupby('date').agg(np.sum)
# ds_validate = ds_temporal_noAgg.drop(row_learn).groupby('date').agg(np.sum)

# baghdad = ds_temporal_noAgg['Adhamiya'].groupby('date').agg(np.sum)
# All_districts = ds_temporal.groupby('date').agg(np.sum).sum(axis=1)
# P = baghdad/All_districts
# Z = 2*np.arcsin(((baghdad+0.25)/(All_districts+0.5))**0.5)
# df_baghdad = baghdad.join(All_districts).join(P).join(Z,how='outer')

# ax = baghdad.plot(figsize=(40,40),color='red')
# ax.set_xticks(xticklocs)
# ax.set_xticklabels(xticklabs,fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.xticks(rotation=50)
# plt.xlabel('(year, month, day)', fontsize=60, family='times new roman')
# plt.ylabel('Number of memos', fontsize=60, family='times new roman')
# plt.show()
# plt.figure()
# autocorrelation_plot(baghdad,color='red')
# 
        




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
    
    fig2 = plt.figure()
    aggregateYearMonth['log'].plot(kind="bar",color=col)
    plt.xticks(rotation=50)
    plt.xlabel('(year, month)', fontsize=40)
    plt.ylabel('Frequency', fontsize=40)
    plt.title('Number of events', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=10)
    fig2 = plt.gcf()
    fig2.set_size_inches(18.5,10.5)
    hist_name = 'eventCount_separate%s'%filename
    plt.savefig(hist_name)         
    plt.close(fig2)
    

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
 
        
def groupAndAggregate(data):
    yr = pd.DatetimeIndex(data['date']).year
    mn = pd.DatetimeIndex(data['date']).month
    dy = pd.DatetimeIndex(data['date']).day
    grouped = data.groupby([yr, mn, dy])
    aggregated = grouped.agg(np.sum)
    return grouped, aggregated
        
def form_distric_df(district,aggregate_data):
    df = aggregate_data[district].to_frame(name=district)
    df['All'] = aggregate_data.sum(axis=1)
    df['Fraction'] = df[district]/df['All']
    df['NormalTrans'] = 2*np.arcsin(((df[district]+0.25)/(df['All']+0.5))**0.5)
    return df
    
     
if __name__ == '__main__':
    main()
