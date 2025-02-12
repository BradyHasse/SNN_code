import numpy as np

w_steps = np.linspace(0.001, 0.035, num=20).tolist()

numUnits = 78

#thresholds = np.linspace(0.0, 0.5, num=20).tolist() #[0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
slopesW4MonkC = [0.19401348, 0.30850939, 0.23415761, 0.10993228, 0.43510231, 0.04693501, 0.04905402, 0.10954162, 0.40073764, 0.12070232,0.05393362, 0.02640521, 0.21301616, 0.02927811, 0.06651825,0.02238517, 0.21540685, 0.10211742, 0.1161748 , 0.13954582,0.20371332, 0.17211011, 0.08040707, 0.07439363, 0.3024469 ,0.51354074, 0.08786162, 0.44560824, 0.04139607, 0.18085206,0.04379441, 0.58378646, 0.20455174, 0.06106598, 0.61508065,0.07574733, 0.02944586, 0.08156803, 0.23931328, 0.01887588,0.01969583, 0.06353518, 0.10951975, 0.35114286, 0.10453103,0.27116546, 0.13000551, 0.65993809, 0.03500946, 0.08120575,0.51861114, 0.42398155, 0.05636304, 0.2203825 , 0.43588027,0.27068338, 0.10614126, 0.03287649, 0.29642382, 0.13091832,0.1389375 , 0.08028673, 0.26314568, 0.21977253, 0.05599628,0.23613622, 0.01942684]

slopesW4MonkN = [0.46018302, 0.09533469, 0.52427498, 0.10134658, 0.06814447,0.16220584, 0.27965731, 0.30781256, 0.59478299, 0.62698187,0.76277736, 1.47242911, 0.5386918 , 0.25916438, 0.0571596 , 0.05087924, 0.18781712, 0.21682462, 0.37628543, 0.19085907,0.01892255, 0.07953764, 0.23825523, 0.20183682, 0.11828916, 0.39950681, 0.38959118, 0.02711883, 0.07185855, 0.24886906, 0.0721993 , 0.52506322, 0.21868481, 0.12361669, 0.07918899,0.08376601, 0.15506244, 0.06300287, 0.30522643, 0.0749404 , 0.37954069, 1.26453004, 0.53973275, 0.07566684, 0.07646446,0.27647286, 0.2600171 , 0.77891962, 0.0341158 , 0.05197552,0.40060392, 0.49568956, 0.6980215 , 0.18372937, 0.5573465 ,0.71463945, 0.32569133, 0.0634245 , 0.36566641, 0.3408889 ,                                                        0.18608109, 0.00732951, 0.25427053, 0.04879866, 0.39242285,  0.07240435, 0.05687793, 0.08007053, 0.05938618, 0.75250851,  0.11423064, 0.09504917, 0.26611557, 0.02579174, 0.04197267, 0.06866484, 0.02212078, 0.02704695]

#units = [x for x in range(0, numUnits)]

units = [58]

unitErrors = []

unitBestValues = []

scaleOut = [None]*numUnits

bestThresholds = []
bestSteps = []

# Should be same as specified in the findBestScale code
steps = np.linspace(0.03, 0.09, num=8)
num_steps = len(steps)

thresholds = np.linspace(0.1, 0.6, 5)


for unit in units:
	all_rmse = []
	all_rmse_first20 = []

	bestValues = []

	minError = 100000000000;
	bestTh = -10
	bestStep = -10
	bestParam = -10

	paramnum=0
	try:
		for th in thresholds:
			for ws1 in range(num_steps):
				for ws2 in range(num_steps):
					for ws3 in range(num_steps):
						# !!!! Please make sure to enter the correct folder name where the results are !!!	
						data1 = np.load(f'RMSE_MonkC_Feb_9_24_Results/RMSE_Unit_{unit}_wStep0_{steps[ws1]}_wStep1_{steps[ws2]}_wStep2_{steps[ws3]}_th_{th}.npz')
						
						items = data1.files
						mean_rmse_first20 = data1[items[1]].tolist()
						print(f'RMSE_Unit_{unit}_wStep0_{steps[ws1]}_wStep1_{steps[ws2]}_wStep2_{steps[ws3]}_th_{th} {mean_rmse_first20}')
						if(mean_rmse_first20 < minError):
							bestTh = th
							bestStep = [steps[ws1], steps[ws2], steps[ws3], th, mean_rmse_first20]
							bestParam = paramnum
							minError = mean_rmse_first20
	#						print(f' Min RMSE results: Unit {unit}, Th: {bestTh} , W0: {bestStep[0]}, W1: {bestStep[1]} , W2: {bestStep[2]}, param: {bestParam}, Error: {minError}')

						minVal = minError
						paramnum = paramnum+1
		bestValues.append(bestStep)
#					bestValues.append(bestTh)
#					bestValues.append(minVal)

		unitBestValues.append(bestStep)
#					print(f' paramnum: {paramnum}')

#					if paramnum == 0 or paramnum == 1794:
#						print(f'paramnum: {paramnum} th: {th}, W0: {steps[ws1]}, W1: {steps[ws2]}, W2: {steps[ws3]}, meanError: {mean_rmse_first20} ')
					
		print(f' Min RMSE results: Unit {unit}, Th: {bestTh} , W0: {bestStep[0]}, W1: {bestStep[1]} , W2: {bestStep[2]}, param: {bestParam}, Error: {minVal}')

	except:
		print(f'File does not exist')
					


## Save the results to some file
with open('TestFilesOut.npy', 'wb') as f:
	np.save(f, unitBestValues)



