"""Click the tool tip + a few points down the SHAFT (for robust multi-point tracking).
Run in your OWN terminal (opens a GUI window):
    <python> click_tooltip_multi.py [frame_path] [out_json]
FIRST click = the tool TIP. Then click 2-3 points down the shaft toward where it
enters the frame. Close the window to save. Order matters: tip first.
"""
import sys, os, json
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    raise SystemExit("Usage: click_tooltip_multi.py <frame_path> [out_json]")
frame_path = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "click_seed_multi.json")

img = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
pts = []

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(img)
ax.set_title(f"Click TIP first, then 2-3 pts down the SHAFT. Close to save. ({W}x{H})")

def onclick(e):
    if e.xdata is None:
        return
    x, y = int(round(e.xdata)), int(round(e.ydata))
    pts.append([x, y])
    color = "lime" if len(pts) == 1 else "yellow"
    ax.plot(e.xdata, e.ydata, "+", color=color, markersize=20, markeredgewidth=3)
    ax.text(e.xdata + 6, e.ydata, f"{len(pts)-1}", color=color, fontsize=12)
    ax.set_title(f"{len(pts)} pts (tip=green). Close to save.")
    fig.canvas.draw()
    print(f"pt {len(pts)-1}: ({x},{y})" + ("  <- TIP" if len(pts) == 1 else ""))

fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()

if pts:
    json.dump({"points": pts, "W": W, "H": H, "frame": 0}, open(out_json, "w"))
    print("SAVED ->", out_json, pts)
else:
    print("No clicks; nothing saved.")
