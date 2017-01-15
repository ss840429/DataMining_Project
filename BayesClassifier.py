import pandas
import math
import numbers
from math import e as exp
import matplotlib
import matplotlib.pyplot as plt
import numpy


def normalDistribution( train, attribute_name, testdata ):

	mean = train[train['quality']==testdata.get('quality')][attribute_name].mean()
	var  = train[train['quality']==testdata.get('quality')][attribute_name].var()

	print( mean, var)



train = pandas.read_csv('wineQuality_Class2_Train.csv')
test  = pandas.read_csv('wineQuality_Class2_Test.csv')



attribute_name = list(train)

for idx, row in test.iterrows():
	
	for attr in attribute_name:
		normalDistribution( train, attr, row)







