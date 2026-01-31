import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0,20,0.1)
f = lambda x, k: 1/(1+x/k)

ks = [2,3,4]
plt.figure(figsize=(8,6))
for k in ks:
    y = f(x,k)
    plt.plot(x,y,label=f'k={k}')
plt.title('Non-linear Depth Function with Varying k')
plt.xlabel('Input x')
plt.ylabel('Output f(x)')
plt.legend()
plt.grid(True)
plt.show()