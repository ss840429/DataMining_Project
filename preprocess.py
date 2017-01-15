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

for idx, d in enumerate(density) :			## Change ph to class 0 or 1 ( sparse or dense )

	if d < 1 :
		density[idx] = 0
	else :
		density[idx] = 1



ph = dataSet['pH']

for idx, p in enumerate(ph) :			## Change ph to class 0 or 1 ( Spicy or Common )

	if p < 3 :
		ph[idx] = 0
	else :
		ph[idx] = 1


alcohol = dataSet['alcohol']

for idx, a in enumerate(alcohol) :			## Change alcohol to class 0 or 1 ( weak or strong )

	if a < 9 :
		alcohol[idx] = 0
	else :
		alcohol[idx] = 1


dataSet.to_csv('wineQuality_Class2.csv', index=False)
dataSet[:rows].to_csv('wineQuality_Class2_Train.csv', index=False)
dataSet[rows:].to_csv('wineQuality_Class2_Test.csv', index=False)

print(dataSet)




