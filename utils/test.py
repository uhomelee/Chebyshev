import numpy.polynomial.chebyshev as chebyshev
import numpy as np
import numpy.linalg as linalg

N=20
x = np.arange(1, N + 1, 1)
A1=A2 = chebyshev.chebvander(x, N - 1)
y1=np.array([0.3230213,0.3259532, 0.3294488, 0.3234509, 0.3262547, 0.3266671, 0.3262381,
 0.3262544, 0.3244484,0.3187518, 0.3187518, 0.3187518, 0.3187518, 0.3187518,
 0.3187518, 0.3187518, 0.3187518, 0.3187518, 0.3187518, 0.3187518,])
y2=np.array([0.3230213, 0.3259532, 0.3294488, 0.3234509, 0.3262547, 0.3266671, 0.3262381,
 0.3262544, 0.3244484, 0.3187518, 0.2940618, 0.2693459, 0.2582519, 0.2511204,
 0.2501969, 0.2451692, 0.2383491, 0.230648,  0.2189554, 0.1783227])
c1=linalg.solve(A1, y1)
c2=linalg.solve(A2,y2)
print('input coef\n', c1)
print('output coef\n', c2)
y1_p=np.dot(A1,c1)
y2_p=np.dot(A2,c2)
print('input predict error\n',y1_p-y1)
print('output predict error\n',y2_p-y2)



# x = np.array([1, 2, 3, 4])
# y = np.array([1, 3, 5, 4])
# deg = len(x) - 1
# A = chebyshev.chebvander(x, deg)
# print(A, "# A")
# c = linalg.solve(A, y)
# print(c,"# c")
# for v in x:
#     print(v, np.polynomial.Chebyshev(c)(v),"#p(%d)" % v)
# print(np.dot(A,c))
# x = np.arange(1,25,1)
# print(x.reshape(2,3,4))
# print(x.reshape(2,3,4).reshape(-1,4).transpose())
# print(x.reshape(2,3,4).reshape(-1,4).transpose().transpose().reshape(-1,3,4))
# print(x.reshape(2,3,4).reshape(-1,4).transpose().transpose().reshape(-1,3,4).swapaxes(2,1))

