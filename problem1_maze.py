
import argparse
import csv
import random
import sys
from collections import deque

if "--show" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAIR_LETTERS = "abcdfg"                      # 'e' is reserved for the exit
VALID = set(".#SE") | set(PAIR_LETTERS) | set(PAIR_LETTERS.upper()) | {"e"}
COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#ff7f0e"]
PATH_GOLD = "#f5b800"
PATH_FILL = "#fff3b0"


def is_key(ch):
    return ch.islower()


def is_door(ch):
    return ch.isupper() and ch not in "SE"


def color_of(letter):
    low = letter.lower()
    if low in PAIR_LETTERS:
        return COLORS[PAIR_LETTERS.index(low)]
    return "#7f7f7f"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_grid_csv(path):
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise ValueError("CSV file is empty")
    grid = []
    for r in rows:
        row = []
        for cell in r:
            if cell == "":
                cell = "."
            # accept a previously written solution file
            if cell.startswith("x"):
                cell = cell[1:].lstrip(":") or "."
            if len(cell) != 1 or cell not in VALID:
                raise ValueError(f"Invalid cell {cell!r} in {path}")
            row.append(cell)
        grid.append(row)
    n = len(grid)
    if any(len(r) != n for r in grid):
        raise ValueError(f"Grid must be square (N x N); got {n} rows, "
                         f"row lengths {sorted({len(r) for r in grid})}")
    flat = [c for r in grid for c in r]
    if flat.count("S") != 1 or flat.count("E") != 1:
        raise ValueError("Grid must contain exactly one 'S' and one 'E'")
    return grid


def write_grid_csv(grid, path, route=None):
    out = [row[:] for row in grid]
    if route:
        for (r, c) in route:
            ch = grid[r][c]
            if ch in "SE":
                continue
            out[r][c] = "x" if ch == "." else f"x:{ch}"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)


# --------------------------------------------------------------------------
# Solver: BFS over (row, col, key-bitmask)
# --------------------------------------------------------------------------
def find(grid, ch):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == ch:
                return (r, c)
    raise ValueError(f"no {ch!r} in grid")


def solve(grid):
    """Return the shortest route as a list of (r, c) cells, or None."""
    n = len(grid)
    start = find(grid, "S")
    end = find(grid, "E")
    s0 = (start[0], start[1], 0)
    parent = {s0: None}
    dq = deque([s0])
    while dq:
        state = dq.popleft()
        r, c, mask = state
        if (r, c) == end:
            route = []
            while state is not None:
                route.append((state[0], state[1]))
                state = parent[state]
            return route[::-1]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            ch = grid[nr][nc]
            if ch == "#":
                continue
            nmask = mask
            if is_door(ch) and not (mask >> (ord(ch.lower()) - 97)) & 1:
                continue
            if is_key(ch):
                nmask |= 1 << (ord(ch) - 97)
            ns = (nr, nc, nmask)
            if ns not in parent:
                parent[ns] = state
                dq.append(ns)
    return None


# --------------------------------------------------------------------------
# Random maze generator
# --------------------------------------------------------------------------
def generate(n, T, rng):
    if not 1 <= T <= 6:
        raise ValueError("T must be between 1 and 6")
    if n < 5 or n * n < 2 + 2 * T + 4:
        raise ValueError("grid too small for that many keys/doors (use N >= 5)")
    letters = PAIR_LETTERS[:T]
    wall_p = 0.27
    for attempt in range(4000):
        if attempt and attempt % 500 == 0:
            wall_p = max(0.1, wall_p - 0.04)
        grid = [["#" if rng.random() < wall_p else "." for _ in range(n)]
                for _ in range(n)]
        third = max(1, n // 3)
        near = [(r, c) for r in range(third) for c in range(third)]
        far = [(r, c) for r in range(n - third, n) for c in range(n - third, n)]
        s, e = rng.choice(near), rng.choice(far)
        used = {s, e}
        free = [(r, c) for r in range(n) for c in range(n)
                if (r, c) not in used and grid[r][c] == "."]
        if len(free) < 2 * T:
            continue
        picks = rng.sample(free, 2 * T)
        grid[s[0]][s[1]] = "S"
        grid[e[0]][e[1]] = "E"
        for i, ch in enumerate(letters):
            kr, kc = picks[i]
            dr, dc = picks[T + i]
            grid[kr][kc] = ch
            grid[dr][dc] = ch.upper()
        route = solve(grid)
        if route is None:
            continue
        # prefer mazes where the shortest route really has to go through doors
        doors_used = {grid[r][c] for r, c in route if is_door(grid[r][c])}
        need = min(2, T) if attempt < 2000 else (1 if attempt < 3500 else 0)
        if len(doors_used) < need:
            continue
        return grid
    raise RuntimeError("could not generate a solvable maze; try a larger N")


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def build_frames(grid, route):
    """Expand the route into animation frames (with sub-frames for events)."""
    frames = []
    held, picked, opened = set(), set(), set()
    last = len(route) - 1
    for i, (r, c) in enumerate(route):
        ch = grid[r][c]
        base = dict(i=i)
        if is_key(ch) and (r, c) not in picked:
            for k, scale in enumerate((1.0, 0.55, 0.0)):
                if k >= 1:
                    held = held | {ch}
                frames.append(dict(base, held=set(held), picked=set(picked),
                                   opened=set(opened),
                                   anim=("key", (r, c), scale),
                                   msg=f"Picked up key '{ch}'"))
            picked.add((r, c))
        elif is_door(ch) and (r, c) not in opened:
            for phase in (0.0, 0.5, 1.0):
                frames.append(dict(base, held=set(held), picked=set(picked),
                                   opened=set(opened),
                                   anim=("door", (r, c), phase),
                                   msg=f"Unlocking door '{ch}' with key '{ch.lower()}'"))
            opened.add((r, c))
        else:
            frames.append(dict(base, held=set(held), picked=set(picked),
                               opened=set(opened), anim=None,
                               msg="Exit reached!" if i == last else ""))
    frames.extend([frames[-1]] * 6)
    return frames


def lerp_color(c1, c2, t):
    a, b = to_rgb(c1), to_rgb(c2)
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def draw_lock(ax, r, c, color):
    """Small open padlock drawn inside an opened door cell."""
    ax.add_patch(Rectangle((c + 0.3, r + 0.5), 0.4, 0.3, fc=color, ec="none", zorder=4))
    ax.plot([c + 0.36, c + 0.36, c + 0.5, c + 0.64],
            [r + 0.5, r + 0.3, r + 0.17, r + 0.3],
            color=color, lw=1.8, solid_capstyle="round", zorder=4)


def draw(ax, ax_inv, grid, route, fr, letters, total_doors):
    n = len(grid)
    ax.clear()
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    visited = set(route[: fr["i"] + 1])
    anim = fr["anim"]
    for r in range(n):
        for c in range(n):
            ch = grid[r][c]
            fc = "white"
            if ch == "#":
                fc = "black"
            elif ch == "S":
                fc = "#2ecc71"
            elif ch == "E":
                fc = "#e74c3c"
            elif (r, c) in visited:
                fc = PATH_FILL
            if is_door(ch):
                col = color_of(ch)
                if (r, c) in fr["opened"]:
                    fc = PATH_FILL if (r, c) in visited else "white"
                elif anim and anim[0] == "door" and anim[1] == (r, c):
                    fc = lerp_color(col, "white", anim[2])
                else:
                    fc = col
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="#c8c8c8", lw=0.5, zorder=1))
            if ch == "S" or ch == "E":
                ax.text(c + .5, r + .55, ch, ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold", zorder=5)
            elif is_door(ch):
                col = color_of(ch)
                unlocked = (r, c) in fr["opened"] or (
                    anim and anim[0] == "door" and anim[1] == (r, c) and anim[2] >= 0.5)
                if unlocked:
                    draw_lock(ax, r, c, col)
                else:
                    ax.text(c + .5, r + .55, ch, ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold", zorder=5)
            elif is_key(ch) and (r, c) not in fr["picked"]:
                scale = anim[2] if anim and anim[0] == "key" and anim[1] == (r, c) else 1.0
                if scale > 0:
                    ax.add_patch(Circle((c + .5, r + .5), 0.32 * scale, fc=color_of(ch),
                                        ec="white", lw=1, zorder=3))
                    if scale > 0.9:
                        ax.text(c + .5, r + .53, ch, ha="center", va="center",
                                color="white", fontsize=8, fontweight="bold", zorder=5)
    xs = [c + .5 for r, c in route]
    ys = [r + .5 for r, c in route]
    ax.plot(xs, ys, ls=":", color=PATH_GOLD, lw=1.2, alpha=.8, zorder=2)
    ax.plot(xs[: fr["i"] + 1], ys[: fr["i"] + 1], color=PATH_GOLD, lw=3, zorder=3,
            solid_capstyle="round")
    pr, pc = route[fr["i"]]
    ax.add_patch(Circle((pc + .5, pr + .5), 0.26, fc="#222", ec="white", lw=1.5, zorder=6))
    ax.set_title(f"Step {fr['i']}/{len(route) - 1}" + (f"   -   {fr['msg']}" if fr["msg"] else ""),
                 fontsize=11)

    # inventory panel
    ax_inv.clear()
    ax_inv.set_xlim(0, 4)
    ax_inv.set_ylim(len(letters) + 2.2, 0)
    ax_inv.set_aspect("equal", adjustable="box")
    ax_inv.axis("off")
    ax_inv.text(2, 0.4, "Key inventory", ha="center", va="center", fontsize=12,
                fontweight="bold")
    for j, L in enumerate(letters):
        y = j + 1.4
        has = L in fr["held"]
        col = color_of(L)
        ax_inv.add_patch(Circle((1, y), 0.38, fc=col if has else "white",
                                ec=col if has else "#aaaaaa", lw=2,
                                ls="-" if has else "--"))
        ax_inv.text(1, y + .03, L, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if has else "#aaaaaa")
        ax_inv.text(1.7, y, "held" if has else "not yet", va="center", fontsize=10,
                    color="#222" if has else "#999")
    ax_inv.text(2, len(letters) + 1.8,
                f"Doors opened: {len(fr['opened'])}/{total_doors}",
                ha="center", va="center", fontsize=10)


def make_figure(n, n_letters):
    size = max(5.0, n * 0.5)
    fig = plt.figure(figsize=(size + 3.2, size + 0.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[size, 3.0], wspace=0.05)
    return fig, fig.add_subplot(gs[0]), fig.add_subplot(gs[1])


def visualize(grid, route, gif_path, png_path, show):
    n = len(grid)
    letters = sorted({ch.lower() for row in grid for ch in row
                      if is_key(ch) or is_door(ch)})
    total_doors = sum(is_door(ch) for row in grid for ch in row)
    fig, ax, ax_inv = make_figure(n, len(letters))
    if route is None:
        fr = dict(i=0, held=set(), picked=set(), opened=set(), anim=None,
                  msg="No path exists")
        draw(ax, ax_inv, grid, [find(grid, "S")], fr, letters, total_doors)
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        print(f"Saved {png_path}")
        if show:
            plt.show()
        plt.close(fig)
        return
    frames = build_frames(grid, route)
    anim = FuncAnimation(
        fig, lambda k: draw(ax, ax_inv, grid, route, frames[k], letters, total_doors),
        frames=len(frames), interval=220, repeat=True)
    draw(ax, ax_inv, grid, route, frames[-1], letters, total_doors)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"Saved {png_path}")
    anim.save(gif_path, writer=PillowWriter(fps=5), dpi=70)
    print(f"Saved {gif_path}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", help="option 1: read the N x N grid from this CSV file")
    src.add_argument("--generate", type=int, metavar="N",
                     help="option 2: generate a random N x N maze (default 12)")
    ap.add_argument("--seed", type=int, help="random seed for generation")
    ap.add_argument("--out", default="maze_solution.csv", help="solution CSV")
    ap.add_argument("--input-out", default="maze_input.csv", help="where to save the solved grid")
    ap.add_argument("--gif", default="maze_animation.gif")
    ap.add_argument("--png", default="maze_final.png")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--no-viz", action="store_true", help="skip the visualisation")
    args = ap.parse_args()

    if sys.stdin.isatty():
        print("Enter T (number of key/door types, 1-6): ", end="", flush=True)
    line = sys.stdin.readline().strip()
    try:
        T = int(line)
    except ValueError:
        sys.exit("Expected an integer T (1-6) on the first line of standard input")
    if not 1 <= T <= 6:
        sys.exit("T must be between 1 and 6")

    if args.csv:
        grid = read_grid_csv(args.csv)
    else:
        rng = random.Random(args.seed)
        grid = generate(args.generate or 12, T, rng)
        write_grid_csv(grid, args.input_out)
        print(f"Generated maze saved to {args.input_out}")

    route = solve(grid)
    print(len(route) - 1 if route else -1)
    write_grid_csv(grid, args.out, route)
    print(f"Saved {args.out}")
    if not args.no_viz:
        visualize(grid, route, args.gif, args.png, args.show)


if __name__ == "__main__":
    main()
