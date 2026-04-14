import numpy as np
import matplotlib.pyplot as plt

from VisFly.utils.FigFashion.FigFashion import FigFon
FigFon.set_fashion("IEEE")
plt.rcParams["hatch.color"] = "black"
plt.rcParams["hatch.linewidth"] = 1.0

def compute_esdf_rectangle(grid_x, grid_y, center, half_size):
	# Signed distance: positive outside, negative inside the rectangle.
	dx = np.abs(grid_x - center[0]) - half_size[0]
	dy = np.abs(grid_y - center[1]) - half_size[1]
	dx_out = np.maximum(dx, 0.0)
	dy_out = np.maximum(dy, 0.0)
	outside = np.sqrt(dx_out * dx_out + dy_out * dy_out)
	inside = np.minimum(np.maximum(dx, dy), 0.0)
	return outside + inside


def main():
	# Grid setup
	x_min, x_max, y_min, y_max = -3.0, 2.0, -1.0, 1.0
	resolution = 0.01
	xs = np.arange(x_min, x_max + resolution, resolution)
	ys = np.arange(y_min, y_max + resolution, resolution)
	grid_x, grid_y = np.meshgrid(xs, ys)

	# Single rectangle obstacle
	center = (1, -0)
	half_size = (0.3, 0.6)
	esdf = compute_esdf_rectangle(grid_x, grid_y, center, half_size)
	esdf_outside = np.ma.array(esdf, mask=esdf <= 0.0)
	grad_y, grad_x = np.gradient(esdf, resolution, resolution)
	grad_x = np.ma.array(grad_x, mask=esdf <= 0.0)
	grad_y = np.ma.array(grad_y, mask=esdf <= 0.0)

	fig = plt.figure(figsize=(6, 3), dpi=600)
	ax = fig.add_subplot(1, 1, 1)
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

	# Draw the rectangle boundary
	lower_left = (center[0] - half_size[0], center[1] - half_size[1])
	# fill_rect = plt.Rectangle(
	# 	lower_left,
	# 	2.0 * half_size[0],
	# 	2.0 * half_size[1],
	# 	fill=True,
	# 	facecolor="white",
	# 	edgecolor="none",
	# 	linewidth=0.0,
	# 	zorder=5,
	# )
	# ax.add_patch(fill_rect)
	# rect = plt.Rectangle(
	# 	lower_left,
	# 	2.0 * half_size[0],
	# 	2.0 * half_size[1],
	# 	fill=False,
	# 	edgecolor="black",
	# 	linewidth=1.5,
	# 	zorder=7,
	# )
	# ax.add_patch(rect)
	hatch_rect = plt.Rectangle(
		lower_left,
		2.0 * half_size[0],
		2.0 * half_size[1],
		fill=False,
		edgecolor="black",
		linewidth=0.8,
		hatch="//////",
		# zorder=6,
	)
	ax.add_patch(hatch_rect)

	ax.set_aspect("equal")
	ax.set_title("")
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlabel("")
	ax.set_ylabel("")

	fig.savefig(
		"/home/lfx-desktop/files/avoidance/debug/esdf_rectangle.png",
		dpi=600,
		bbox_inches="tight",
		pad_inches=0,
		facecolor="white",
	)

if __name__ == "__main__":
    main()
