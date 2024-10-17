import ydf
import pandas as pd

pdA = pd.DataFrame(A)
pdA.columns = ['p1', 'p2', 'p3', 'p4']
pdA.insert(0, 'p0', [int(round(x)) for x in Cl2])

pdA_test = pd.DataFrame(A_test)
pdA_test.columns = ['p1', 'p2', 'p3', 'p4']
pdA_test.insert(0, 'p0', [int(round(x)) for x in Cl2_test])

model = ydf.GradientBoostedTreesLearner(label="p0", task=ydf.Task.REGRESSION).train(pdA)
model = ydf.GradientBoostedTreesLearner(label="p0", task=ydf.Task.CLASSIFICATION).train(pdA)

model.describe()
model.evaluate(pdA_test)
model.predict(pdA_test)
model.analyze(pdA_test)
model.benchmark(pdA_test)
