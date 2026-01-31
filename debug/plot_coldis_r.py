import torch as th


k = 0.015
func = lambda x: 2.4 * k / (x + k)
func3 = lambda x: 2.5 * th.log(1 + th.exp(-32 * x))

x = th.arange(0, 1.0, 0.01)
y1 = func(x)
y2 = func3(x)

import matplotlib.pyplot as plt
from VisFly.utils.FigFashion.FigFashion import FigFon
FigFon.set_fashion("IEEE")
fig, ax = FigFon.get_figure_axes(
    SubFigSize=(1, 1),
    HeightScale=1,
    Column=1,
    # FigName="Collision Distance vs. Proximity Value",
)

ax.plot(x.numpy(), y1.numpy(), label="Rational Function", linewidth=2)
ax.plot(x.numpy(), y2.numpy(), label="Softplus Function", linewidth=2)

ax.set_xlabel("Proximity Value (m)")
ax.set_ylabel("Collision Distance (m)")
ax.set_title("Collision Distance vs. Proximity Value")
ax.legend()
plt.show()

