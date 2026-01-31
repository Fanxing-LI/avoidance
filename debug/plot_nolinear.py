import matplotlib.pyplot as plt
import numpy as np

func = lambda x, k : k / (x+k) * 2
f2 = lambda x: 2.5 * np.log(1+np.exp(-32*x))

x = np.linspace(0, 1, 100)

k = [
    # 1,
     # 0.5,
     # 0.15,
     # 0.12,
     # 0.1,
     0.05,
     0.02,
     0.015,
     0.01,
     0.005,
     0.002
     ]

fig, ax = plt.subplots()
for i in k:
    ax.plot(x, func(x, i), label=f"k={i}")
ax.plot(x, f2(x), label="f2", linestyle="--", color="black")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.show()
