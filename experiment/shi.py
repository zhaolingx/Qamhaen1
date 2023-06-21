import numpy as np
Array=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
print(Array)
Array=Array.reshape(4,3,-1)
print(Array)
Array=Array.reshape(2,-1)
print(Array)