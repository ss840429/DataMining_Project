import pandas
import numpy


dataSet = pandas.read_csv('wineQuality.csv')
dataSet.info()
rows = int(len(dataSet.index)*5/6)

quality = dataSet['quality']

for idx, q in enumerate(quality) :		## Change quality to class 0 or 1 ( Not good or Good )
	
	if q < 6 :
		quality[idx] = 0
	else :
		quality[idx] = 1


dataSet.to_csv('wineQuality_Class1.csv', index=False)

density = dataSet['density']
chlorides = dataSet['chlorides']
volatile = dataSet['volatile acidity']


for idx, (d, p, a) in enumerate(zip(density, chlorides, volatile)) :			## Normalize 

	density[idx] = (d - density.min()) / (density.max() - density.min())
	chlorides[idx] 	 = (p - chlorides.min()) / (chlorides.max() - chlorides.min())
	volatile[idx] = (a - volatile.min()) / (volatile.max() - volatile.min())


dataSet.to_csv('wineQuality_Class2.csv', index=False)

print(dataSet)


