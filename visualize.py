import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Generate a simple synthetic cube and annotations
T, D, H, W = 20, 16, 64, 64
cube = np.zeros((T, D, H, W), dtype=np.float32)
annotations = []

y0, x0 = H/4, W/4
vy, vx = 1.0, 0.5
d0, dv = 4, 0.2
r_mask = 3

# Fill cube and annotations
true_d = []
for t in range(T):
    y = np.clip(y0 + vy * t, 0, H - 1)
    x = np.clip(x0 + vx * t, 0, W - 1)
    d = int(np.clip(d0 + dv * t, 0, D - 1))
    true_d.append(d)
    yy, xx = np.ogrid[:H, :W]
    mask2d = ((yy - y)**2 + (xx - x)**2) <= r_mask**2
    cube[t, d][mask2d] = 1.0
    annotations.append((t, d, mask2d))

# 1) Visualize A-R for first 6 time slices
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for idx in range(6):
    ax = axes[0, idx]
    img2d = cube[idx].max(axis=0)  # project D dimension
    ax.imshow(img2d, cmap='viridis')
    t, d, mask2d = annotations[idx]
    ys, xs = np.where(mask2d)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red')
    ax.add_patch(rect)
    ax.set_title(f"t={idx}, d={d}")
    ax.axis('off')
axes[0, 0].set_ylabel("R")

# 2) Visualize T vs D (x = D, y = T)
heat = cube.sum(axis=(2, 3))  # [T, D]
ax = axes[1, 0]
im = ax.imshow(heat, aspect='auto', origin='upper', cmap='plasma')
ax.set_title("T vs D")
ax.set_xlabel("D Bin")
ax.set_ylabel("T")
# overlay true trace: x=true_d, y=times
times = np.arange(T)
ax.plot(true_d, times, color='cyan', marker='o', markersize=3, linewidth=1.5, label='True d')
ax.legend()

# hide unused subplots
for j in range(1, 6):
    axes[1, j].axis('off')

fig.colorbar(im, ax=axes[1, 0], location='right', shrink=0.6)
plt.tight_layout()
plt.show()
