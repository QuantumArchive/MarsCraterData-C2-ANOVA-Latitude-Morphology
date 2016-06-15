# -*- coding: utf-8 -*-
"""
Created on Tue June 9 07:49:05 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import statsmodels.formula.api as smf
import scipy.stats

#from IPython.display import display
%matplotlib inline

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with NaN
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ',numpy.NaN)

#'We'll define the region between -30 and 30 to be equatorial and -90 to -30 and 30 to 90 to be at the pole
def georegion(x):
    if x <= -30:
        return 'POLES'
    elif x <= 30:
        return 'EQUATOR'
    else:
        return 'POLES'

data['LATITUDE_BIN'] = data['LATITUDE_CIRCLE_IMAGE'].apply(georegion)
data['LATITUDE_BIN'] = data['LATITUDE_BIN'].astype('category')

print('Let us now look at data with only the top 3 morphology types present')

#slice out the rows with just the morphology we want
morphofinterest = ['Rd', 'SLEPS', 'SLERS']
data = data.loc[data['MORPHOLOGY_EJECTA_1'].isin(morphofinterest)]

#We'll now subset out the columns we're interested in
latitude = numpy.array(data['LATITUDE_BIN'])
morphology = numpy.array(data['MORPHOLOGY_EJECTA_1'])
diameter = numpy.array(data['DIAM_CIRCLE_IMAGE'])
data2 = pandas.DataFrame({'LATITUDE_BIN':latitude,'MORPHOLOGY_EJECTA_1':morphology,'DIAM_CIRCLE_IMAGE':diameter}).dropna()

#we now look at the statistics for the key value of Latitude Bin and Morphology Ejecta
data3 = data2.groupby(['LATITUDE_BIN','MORPHOLOGY_EJECTA_1']).mean()
data3.rename(columns={"DIAM_CIRCLE_IMAGE":"DIAM_CIRCLE_IMAGE_MEAN"},inplace=True)
data4 = data2.groupby(['LATITUDE_BIN','MORPHOLOGY_EJECTA_1']).std()
data4.rename(columns={"DIAM_CIRCLE_IMAGE":"DIAM_CIRCLE_IMAGE_STDEV"},inplace=True)
data5 = pandas.concat([data3,data4],axis=1)
data5

#We now plot the mean and standard deviation for the crater diameter for each morphology type located between the poles
#and the equator
gplot = seaborn.factorplot(x='LATITUDE_BIN',y='DIAM_CIRCLE_IMAGE',data=data2,col='MORPHOLOGY_EJECTA_1',kind='bar')
gplot

#now we loop through the ANOVA analysis for craters found at either the Poles or the Equator for different selecta ejecta
#morphology
for a0 in morphofinterest:
    tempdata = data2.loc[data2['MORPHOLOGY_EJECTA_1']==a0]
    tempmodel = smf.ols(formula='DIAM_CIRCLE_IMAGE ~ C(LATITUDE_BIN)',data=tempdata)
    tempresults = tempmodel.fit()
    print('ANOVA STUDY :' + a0)
    print(tempresults.summary())