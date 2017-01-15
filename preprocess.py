import pandas



dataSet = pandas.read_csv('wineQuality.csv')
dataSet.info()

quality = dataSet['quality']

for idx, q in enumerate(quality) :		## Change quality to class 0 or 1 ( Not good or Good )
	
	if q < 5 :
		quality[idx] = 0
	else :
		quality[idx] = 1


dataSet.to_csv('wineQuality_Class1.csv')


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


dataSet.to_csv('wineQuality_Class2.csv')

print(dataSet)