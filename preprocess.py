import pandas


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
dataSet[:rows].to_csv('wineQuality_Class1_Train.csv', index=False)
dataSet[rows:].to_csv('wineQuality_Class1_Test.csv', index=False)

density = dataSet['density']
ph      = dataSet['pH']
alcohol = dataSet['alcohol']



for idx, (d, p, a) in enumerate(zip(density, ph, alcohol)) :			## Change density, ph and alcohol to class 0 or 1 

	density[idx] = False if d < 0.995 else True 
	ph[idx] 	 = False if p < 3 	  else True	
	alcohol[idx] = False if a < 9 	  else True



dataSet.to_csv('wineQuality_Class2.csv', index=False)
dataSet[:rows].to_csv('wineQuality_Class2_Train.csv', index=False)
dataSet[rows:].to_csv('wineQuality_Class2_Test.csv', index=False)

print(dataSet)


