import numpy as np
import matplotlib.pyplot as plt

from VisFly.utils.FigFashion.FigFashion import FigFon
FigFon.set_fashion("IEEE")
plt.rcParams["hatch.color"] = "black"
plt.rcParams["hatch.linewidth"] = 1.0

def compute_esdf_circle(grid_x, grid_y, center, radius):
    # Signed distance: positive outside, negative inside the circle.
    dx = grid_x - center[0]
    dy = grid_y - center[1]
    return np.sqrt(dx * dx + dy * dy) - radius


def main():
	# Grid setup
	x_min, x_max, y_min, y_max = -3.0, 2.0, -1.0, 1.0
	resolution = 0.01
	xs = np.arange(x_min, x_max + resolution, resolution)
	ys = np.arange(y_min, y_max + resolution, resolution)
	grid_x, grid_y = np.meshgrid(xs, ys)

	# Single circle obstacle
	center = (1, -0.0)
	radius = 0.3
	esdf = compute_esdf_circle(grid_x, grid_y, center, radius)
	esdf_outside = np.ma.array(esdf, mask=esdf <= 0.0)
	grad_y, grad_x = np.gradient(esdf, resolution, resolution)
	grad_x = np.ma.array(grad_x, mask=esdf <= 0.0)
	grad_y = np.ma.array(grad_y, mask=esdf <= 0.0)

	fig, ax = FigFon.get_figure_axes(SubFigSize=(1, 1), Column=1, share_legend=False, HeightScale=0.8)
	fig.patch.set_facecolor("white")
	ax.set_facecolor("white")
	im = ax.contourf(
		grid_x,
		grid_y,
		esdf_outside,
		levels=60,
		cmap="viridis",
		alpha=0.8,
		antialiased=False,
        linewidths=3
	)
	step = 18
	ax.quiver(
		grid_x[::step, ::step],
		grid_y[::step, ::step],
		grad_x[::step, ::step],
		grad_y[::step, ::step],
		color="black",
		angles="xy",
		scale_units="xy",
		scale=15.0,
		width=0.002,
		alpha=0.9,
	)
	ax.contour(grid_x, grid_y, esdf, levels=[0.0], colors="black", linewidths=1.5)

	# Draw the circle boundary
	circle = plt.Circle(center, radius, fill=False, color="black", linewidth=1.5)
	ax.add_patch(circle)
	hatch_circle = plt.Circle(
		center,
		radius,
		fill=False,
		edgecolor="black",
		linewidth=0.8,
		hatch="//////",
	)
	ax.add_patch(hatch_circle)

	ax.set_aspect("equal")
	ax.set_title("")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel("")
	ax.set_ylabel("")

	fig.savefig(
		"/home/lfx-desktop/files/avoidance/debug/esdf_circle.png",
		dpi=600,
		bbox_inches="tight",
		pad_inches=0,
		facecolor="white",
	)

if __name__ == "__main__":
    main()
