import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import plotly.plotly as py
import plotly.figure_factory as ff
plt.rcdefaults()


#declare Global
finalpredicted = []
statesList = []
total_percE = []
total_percP = []
tempListP = []
df_16List = []
df_17List = [] 

#Read All CSV
#Make States Dataframe
states = pd.read_csv("StatesN.csv")
states = states.iloc[:,0]
statesList = list(states)

#Make Energy DataFrame
dfE = pd.read_csv("book1.csv")
df_total = dfE.iloc[:,-2]
totalE = list(df_total)
df_statesE = dfE.iloc[:,0]
df_statesE = list(df_statesE)

#Make Population DataFrame
########################################################################
dfP = pd.read_csv("book2C.csv")
df_totalP = dfP.iloc[:,1]
totalP = list(df_totalP)
df_statesP = dfP.iloc[:,0]
df_statesP = list(df_statesP)
#########################################################################

#Calclate Percnetage Increase
########################################################################
def perc(x,listD,listS):
    total_array = np.array(listD,dtype=float)
    ix = listS.index(x)
    total_x_array = total_array[ix:ix+5]
    listFinal = []
    for i in range(4):
        temp = (total_x_array[i+1]-total_x_array[i])*100/total_x_array[i]
        listFinal.append(temp)
    return listFinal
#######################################################################
    
#Machine Learning Algorithm
#######################################################################
def evaluate_algorithm(dataset, algorithm, train,test):
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
        predicted = algorithm(train, test_set)
        return predicted
        
def mean(values):
	return sum(values) / float(len(values))
 
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
 
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
 
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]
 
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions    
#######################################################################
#All Functions To Make Data For Machine
########################################################################
def initializeList():
    for i in range(3):
        train.append([])
        test.append([])
        dataset.append([])

def makeTrainData():
    for i in range(3):
        dataset[i].append(x[i])
        dataset[i].append(y[i])
        train[i].append(x[i])
        train[i].append(y[i])
    
def makeTestData():
    for i in range(3):
        test[i].append(x[3])
        test[i].append(y[3])
    return y[3]
#######################################################################
#Creating 2017-16 Energy List
########################################################################
df_2016E = dfE.iloc[:,-2]
df_Elist = list(df_2016E)
def E2016(i,df_EList,df_statesE):
    ix = df_statesE.index(i)
    temp = df_EList[ix+3]
    return temp;

def E2017(i,df_EList,df_statesE):
    ix = df_statesE.index(i)
    temp = df_EList[ix+4]
    return temp;

for i in statesList:
    temp = E2017(i,df_Elist,df_statesE)
    df_17List.append(temp)
    
for i in statesList:
    temp = E2016(i,df_Elist,df_statesE)
    df_16List.append(temp)
df_16A = np.array(df_16List)
########################################################################
#Implementing Machine Learning
#####################################################################
for i in statesList:
    total_percE = perc(i,totalE,df_statesE)
    total_percP = perc(i,totalP,df_statesP)
    dataset = []
    test = []
    train = []
    tempList = []
    x = list(total_percP)
    y= list(total_percE)
    initializeList()
    makeTrainData()
    tempListP.append(makeTestData())
    tempList = evaluate_algorithm(dataset, simple_linear_regression, train,test)
    finalpredicted.append(tempList[0])
finalarray = np.array(finalpredicted)
finalarray = finalarray+100
temp_array = finalarray * df_16A
temp_array = temp_array / 100
Predicted_2017E = list(temp_array)
#############################################################################
#Predictions to be Displayed on Map 2017
##############################################################################
def predict():
    df_predict_display = pd.DataFrame({'State':statesList,'2017A':df_17List,'2017P':Predicted_2017E})
    df_predicted2017 = pd.DataFrame({'State':statesList,'2017':Predicted_2017E })
    fig, ax = plt.subplots(figsize=(10,20))
    india = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=27.90, lon_0=29.4,
            llcrnrlon=68.03, llcrnrlat= 7.78, urcrnrlon=97.43, urcrnrlat=35.76)
    india.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    india.drawmapboundary(fill_color='lightblue')
    india.fillcontinents(color='yellow',lake_color='#46bcec')
    india.drawcoastlines()
    india.readshapefile('F:\SDLProjects\Shapefile\India','areas')
    df_poly2017Predicted = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in india.areas],
        'State': [area['NAME_1'] for area in india.areas_info]})
    df_poly2017Predicted = df_poly2017Predicted.merge(df_predicted2017 , on ='State' , how = 'left')
    cmap = plt.get_cmap('Oranges')
    pc=PatchCollection(df_poly2017Predicted.shapes,zorder=2)
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_poly2017Predicted['2017'].fillna(0).values)))
    ax.add_collection(pc)
    mapper = matplotlib.cm.ScalarMappable(norm=norm , cmap=cmap)
    mapper.set_array(df_poly2017Predicted['2017'])
    plt.colorbar(mapper,shrink=0.4)
    perIncreaseP = list(finalpredicted)
    perIncreaseA = list(tempListP)
    lenght = len(finalpredicted)
    fig, ax = plt.subplots()
    ind = np.arange(lenght)
    bar_width = 0.35
    rects1 = plt.bar(ind,perIncreaseP,bar_width,color='gold',label='Predicted%')
    rects2 = plt.bar(ind + bar_width,perIncreaseA,bar_width,color='yellowgreen',label='Actual%')
    plt.xlabel('States')
    plt.ylabel('Energy Growth in %')
    plt.xticks(ind + bar_width,())
    plt.legend()
    plt.tight_layout()
    print(df_predict_display)
    plt.show()
##############################################################################
#Actual Dataframe for 2017 Energy
###############################################################################
def actual():
    df_actual2017 = pd.DataFrame({'State':statesList,'2017':df_17List})
    fig, ax = plt.subplots(figsize=(10,20))
    india = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=27.90, lon_0=29.4,
            llcrnrlon=68.03, llcrnrlat= 7.78, urcrnrlon=97.43, urcrnrlat=35.76)
    india.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    india.drawmapboundary(fill_color='lightblue')
    india.fillcontinents(color='yellow',lake_color='#46bcec')
    india.drawcoastlines()
    india.readshapefile('F:\SDLProjects\Shapefile\India','areas')
    df_poly2017Actual = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in india.areas],
        'State': [area['NAME_1'] for area in india.areas_info]})
    df_poly2017Actual = df_poly2017Actual.merge(df_actual2017 , on ='State' , how = 'left')
    cmap = plt.get_cmap('Oranges')
    pc=PatchCollection(df_poly2017Actual.shapes,zorder=2)
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_poly2017Actual['2017'].fillna(0).values)))
    ax.add_collection(pc)
    mapper = matplotlib.cm.ScalarMappable(norm=norm , cmap=cmap)
    mapper.set_array(df_poly2017Actual['2017'])
    plt.colorbar(mapper,shrink=0.4)
    print(df_actual2017)
    plt.show()
##############################################################################



