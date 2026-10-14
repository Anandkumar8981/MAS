import argparse
import csv
import math
import random
import sys

from itertools import product

if "--show" not in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

INF = float("inf")
RAD = 0.14
NODE_R = 0.05


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def solve(n, edges, K, C):
    """edges: list of (u, v, p).  Returns None or dict(cost, nodes, flights).

    flights: list of dict(idx, u, v, price, paid, coupon) in travel order.
    """
    if n == 1:
        return dict(cost=0, nodes=[1], flights=[])
    max_f = K + 1
    dp = [[[INF] * (C + 1) for _ in range(n + 1)] for _ in range(max_f + 1)]
    par = {}
    dp[0][1][0] = 0
    for e in range(max_f):
        layer, nxt = dp[e], dp[e + 1]
        for idx, (u, v, p) in enumerate(edges):
            half = p // 2
            for c in range(C + 1):
                cur = layer[u][c]
                if cur == INF:
                    continue
                if cur + p < nxt[v][c]:
                    nxt[v][c] = cur + p
                    par[(e + 1, v, c)] = (u, c, idx, False)
                if c < C and cur + half < nxt[v][c + 1]:
                    nxt[v][c + 1] = cur + half
                    par[(e + 1, v, c + 1)] = (u, c, idx, True)
    best, state = INF, None
    for e in range(1, max_f + 1):          # fewer flights wins ties
        for c in range(C + 1):
            if dp[e][n][c] < best:
                best, state = dp[e][n][c], (e, n, c)
    if state is None:
        return None
    flights = []
    e, v, c = state
    while e > 0:
        u, pc, idx, used = par[(e, v, c)]
        p = edges[idx][2]
        flights.append(dict(idx=idx, u=u, v=v, price=p, paid=p // 2 if used else p, coupon=used))
        e, v, c = e - 1, u, pc
    flights.reverse()
    nodes = [1] + [f["v"] for f in flights]
    return dict(cost=best, nodes=nodes, flights=flights)


def brute_force(n, edges, K, C):
    """Independent check: enumerate every walk with <= K+1 flights; for a walk
    the best coupon use is on the C flights with the largest saving."""
    if n == 1:
        return 0
    adj = {}
    for u, v, p in edges:
        adj.setdefault(u, []).append((v, p))
    best = [INF]

    def dfs(city, used, total, savings):
        if city == n and used > 0:
            cost = total - sum(sorted(savings, reverse=True)[:C])
            best[0] = min(best[0], cost)
            return
        if used == K + 1:
            return
        for v, p in adj.get(city, []):
            dfs(v, used + 1, total + p, savings + [p - p // 2])

    dfs(1, 0, 0, [])
    return best[0] if best[0] < INF else -1


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _int(s, what):
    try:
        f = float(s)
    except ValueError:
        raise ValueError(f"{what}: expected a number, got {s!r}")
    if f != int(f):
        raise ValueError(f"{what}: expected an integer, got {s!r}")
    return int(f)


def validate(n, edges, K, C):
    if n < 1:
        raise ValueError("N must be at least 1")
    if K < 0 or C < 0:
        raise ValueError("K and C must be non-negative")
    for u, v, p in edges:
        if not (1 <= u <= n and 1 <= v <= n):
            raise ValueError(f"flight {u}->{v} uses a city outside 1..{n}")
        if p < 0:
            raise ValueError(f"flight {u}->{v} has negative price {p}")


def read_option1(text):
    tok = text.split()
    try:
        n, m, K, C = (_int(t, "header") for t in tok[:4])
        body = tok[4:4 + 3 * m]
        if len(body) < 3 * m:
            raise ValueError(f"expected {m} flights, found only {len(body) // 3}")
        edges = [(_int(body[3 * i], "u"), _int(body[3 * i + 1], "v"), _int(body[3 * i + 2], "price"))
                 for i in range(m)]
    except IndexError:
        raise ValueError("input too short; expected 'N M', 'K C' and M flights")
    validate(n, edges, K, C)
    return n, edges, K, C


def read_KC(text):
    tok = text.split()
    if len(tok) < 2:
        raise ValueError("expected 'K C' on the first line of standard input")
    return _int(tok[0], "K"), _int(tok[1], "C")


def read_graph_csv(path, fmt="auto", n_override=None):
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise ValueError("CSV file is empty")
    header = any(c and not _is_num(c) for c in rows[0])
    if fmt == "auto":
        square = all(len(r) == len(rows) for r in rows)
        fmt = "edges" if header or not square else "matrix"
    if fmt == "matrix":
        n = len(rows)
        if any(len(r) != n for r in rows):
            raise ValueError("matrix CSV must be square")
        edges = []
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                if c and _int(c, "price") > 0 and i != j:
                    edges.append((i + 1, j + 1, _int(c, "price")))
    else:
        if header:
            rows = rows[1:]
        edges = []
        for r in rows:
            if len(r) < 3:
                raise ValueError(f"edge row needs u,v,price: {r}")
            edges.append((_int(r[0], "u"), _int(r[1], "v"), _int(r[2], "price")))
        n = n_override or max((max(u, v) for u, v, _ in edges), default=1)
    if n_override:
        n = n_override
    return n, edges


def write_edges_csv(path, edges):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "price"])
        w.writerows(edges)


def write_solution_csv(path, edges, sol):
    used = {}
    if sol:
        for fl in sol["flights"]:
            used.setdefault(fl["idx"], fl)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "price", "on_path", "coupon_used", "paid_price"])
        for i, (u, v, p) in enumerate(edges):
            fl = used.get(i)
            if fl:
                w.writerow([u, v, p, "x", "x" if fl["coupon"] else "", fl["paid"]])
            else:
                w.writerow([u, v, p, "", "", ""])


# --------------------------------------------------------------------------
# Random generator
# --------------------------------------------------------------------------
def generate(n, m, K, rng, pmin=20, pmax=200):
    if n < 2:
        raise ValueError("need at least 2 cities")
    max_flights = K + 1
    hi = min(max_flights, n - 1)
    lo = min(2, hi)
    L = rng.randint(lo, hi)
    route = [1] + rng.sample(range(2, n), L - 1) + [n]
    edges = {}
    for a, b in zip(route, route[1:]):
        edges[(a, b)] = rng.randint(pmin, pmax)
    if (1, n) not in edges:
        edges[(1, n)] = rng.randint(2 * pmax, 3 * pmax)          # pricey direct flight
    m = min(max(m, len(edges)), n * (n - 1))
    tries = 0
    while len(edges) < m and tries < 100 * m:
        tries += 1
        u, v = rng.sample(range(1, n + 1), 2)
        if (u, v) not in edges:
            edges[(u, v)] = rng.randint(pmin, pmax)
    items = [(u, v, p) for (u, v), p in edges.items()]
    rng.shuffle(items)
    return items


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def make_layout(n, edges, seed=1):
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from((u, v) for u, v, _ in edges)
    if n == 1:
        return {1: (0.5, 0.5)}
    init = {1: (-1.0, 0.0), n: (1.0, 0.0)}
    rng = random.Random(seed)
    for v in range(2, n):
        init[v] = (rng.uniform(-0.8, 0.8), rng.uniform(-1, 1))
    pos = nx.spring_layout(G, pos=init, fixed=[1, n], seed=seed,
                           k=1.4 / math.sqrt(n), iterations=300)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    sx = (max(xs) - min(xs)) or 1.0
    sy = (max(ys) - min(ys)) or 1.0
    return {v: (0.08 + 0.84 * (p[0] - min(xs)) / sx, 0.08 + 0.84 * (p[1] - min(ys)) / sy)
            for v, p in pos.items()}


def bezier(p0, p2, t, rad=RAD):
    mx, my = (p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    c = (mx + rad * dy, my - rad * dx)                   # matplotlib "arc3" control point
    a, b, d = (1 - t) ** 2, 2 * t * (1 - t), t * t
    return (a * p0[0] + b * c[0] + d * p2[0], a * p0[1] + b * c[1] + d * p2[1])


def draw(ax, axi, n, edges, pos, sol, st, K, C):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    path_idx = {}
    nodes_on_path = set()
    if sol:
        for j, fl in enumerate(sol["flights"]):
            path_idx.setdefault(fl["idx"], j)
        nodes_on_path = set(sol["nodes"])
    circles = {}
    for v in range(1, n + 1):
        if v == 1:
            fc = "#2ecc71"
        elif v == n:
            fc = "#e74c3c"
        elif v in nodes_on_path:
            fc = "#ff9f1c"
        else:
            fc = "#dfe7ef"
        here = st["city"] == v
        circ = Circle(pos[v], NODE_R, fc=fc, ec="black" if here else "#555555",
                      lw=3 if here else 1.2, zorder=5)
        ax.add_patch(circ)
        circles[v] = circ
        ax.text(pos[v][0], pos[v][1], str(v), ha="center", va="center", fontsize=11,
                fontweight="bold", color="white" if fc in ("#2ecc71", "#e74c3c") else "#222",
                zorder=6)

    def arrow(u, v, color, lw, ls, z, alpha=1.0):
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=13,
                                     connectionstyle=f"arc3,rad={RAD}", patchA=circles[u],
                                     patchB=circles[v], shrinkA=0, shrinkB=0, color=color,
                                     lw=lw, ls=ls, zorder=z, alpha=alpha))

    def label(u, v, text, color, z, fs=8, bold=False, bg="white"):
        x, y = bezier(pos[u], pos[v], 0.5)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=z,
                bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none", alpha=0.9))

    for i, (u, v, p) in enumerate(edges):
        if i in path_idx:
            continue
        arrow(u, v, "#b9b9b9", 1.0, "-", 2)
        label(u, v, str(p), "#888888", 3, fs=7)
    done = st["k"]
    for j, fl in enumerate(sol["flights"] if sol else []):
        u, v = fl["u"], fl["v"]
        active = j < done or (j == done and st["t"] is not None)
        if fl["coupon"] and active:
            arrow(u, v, "#f5b800", 4.0, "-", 4)
            label(u, v, f"{fl['price']} → {fl['paid']}", "#7a5c00", 7, fs=9, bold=True,
                  bg="#fff3b0")
        elif active:
            arrow(u, v, "#1f77b4", 3.2, "-", 4)
            label(u, v, str(fl["price"]), "#1f4e79", 7, fs=9, bold=True)
        else:
            arrow(u, v, "#1f77b4", 1.8, (0, (4, 3)), 4)
            label(u, v, str(fl["price"]), "#1f4e79", 7, fs=8)
    if sol and st["t"] is not None and st["k"] < len(sol["flights"]):
        fl = sol["flights"][st["k"]]
        x, y = bezier(pos[fl["u"]], pos[fl["v"]], 0.12 + 0.76 * st["t"])
        ax.add_patch(Circle((x, y), 0.022, fc="#111", ec="white", lw=1.5, zorder=6.5))

    # info panel
    axi.clear()
    axi.axis("off")
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.text(0.02, 0.97, "Journey", fontsize=13, fontweight="bold", va="top")
    axi.text(0.02, 0.89, f"Running cost:  {st['cost']}", fontsize=12, va="top")
    axi.text(0.02, 0.83, f"Stops used:     {st['stops']} / {K}", fontsize=12, va="top")
    axi.text(0.02, 0.77, f"Coupons used: {st['coupons']} / {C}", fontsize=12, va="top")
    if st["msg"]:
        axi.text(0.02, 0.69, st["msg"], fontsize=10.5, va="top", color="#7a5c00",
                 fontweight="bold", wrap=True)
    y = 0.58
    axi.text(0.02, y, "Flights taken", fontsize=10.5, fontweight="bold", va="top")
    for line in st["log"][-9:]:
        y -= 0.058
        axi.text(0.02, y, line, fontsize=9.5, va="top", family="monospace")
    # legend
    handles = [("#2ecc71", "start (city 1)"), ("#e74c3c", f"destination (city {n})"),
               ("#ff9f1c", "intermediate stop")]
    for i, (col, txt) in enumerate(handles):
        yy = 0.12 - i * 0.04
        axi.plot([0.06], [yy], "o", ms=9, mfc=col, mec="#555", transform=axi.transAxes)
        axi.text(0.11, yy, txt, fontsize=8.5, va="center")
    axi.plot([0.03, 0.09], [0.16, 0.16], color="#f5b800", lw=3, transform=axi.transAxes)
    axi.text(0.11, 0.16, "flight with coupon", fontsize=8.5, va="center")


def build_frames(sol, K, C, hold=8, steps=6):
    frames = []
    cost, stops, coupons, log = 0, 0, 0, []
    base = dict(city=1)
    frames.append(dict(base, k=0, t=None, cost=0, stops=0, coupons=0, log=list(log),
                       msg="Start in city 1"))
    nflights = len(sol["flights"])
    for j, fl in enumerate(sol["flights"]):
        last = j == nflights - 1
        for s in range(steps):
            t = s / (steps - 1)
            msg = f"Flying {fl['u']} → {fl['v']}"
            if fl["coupon"]:
                msg += f"\nCoupon applied: {fl['price']} → {fl['paid']}"
            frames.append(dict(city=fl["u"], k=j, t=t, cost=cost, stops=stops,
                               coupons=coupons, log=list(log), msg=msg))
        cost += fl["paid"]
        coupons += fl["coupon"]
        if not last:
            stops += 1
        log.append(f"{fl['u']}→{fl['v']}  " +
                   (f"{fl['price']}→{fl['paid']} *" if fl["coupon"] else f"{fl['price']}"))
        frames.append(dict(city=fl["v"], k=j + 1, t=None, cost=cost, stops=stops,
                           coupons=coupons, log=list(log),
                           msg=f"Landed in city {fl['v']}"))
    final = dict(frames[-1], msg=f"Arrived! Total cost = {cost}")
    frames.extend([final] * hold)
    return frames


def visualize(n, edges, sol, K, C, gif_path, png_path, show):
    pos = make_layout(n, edges)
    size = 7.2
    fig = plt.figure(figsize=(size + 3.6, size))
    gs = fig.add_gridspec(1, 2, width_ratios=[size, 3.6], wspace=0.02)
    ax, axi = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    if sol is None:
        st = dict(city=1, k=0, t=None, cost=0, stops=0, coupons=0, log=[],
                  msg=f"No route with at most {K} stops")
        draw(ax, axi, n, edges, pos, None, st, K, C)
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        print(f"Saved {png_path}")
        if show:
            plt.show()
        plt.close(fig)
        return
    frames = build_frames(sol, K, C)
    anim = FuncAnimation(fig, lambda k: draw(ax, axi, n, edges, pos, sol, frames[k], K, C),
                         frames=len(frames), interval=200, repeat=True)
    draw(ax, axi, n, edges, pos, sol, frames[-1], K, C)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"Saved {png_path}")
    anim.save(gif_path, writer=PillowWriter(fps=6), dpi=65)
    print(f"Saved {gif_path}")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
def self_test(trials=4000, seed=0):
    rng = random.Random(seed)
    bad = 0
    for t in range(trials):
        n = rng.randint(1, 7)
        m = rng.randint(0, 14) if n > 1 else 0
        edges = []
        for _ in range(m):
            u, v = rng.randint(1, n), rng.randint(1, n)
            if u != v:
                edges.append((u, v, rng.choice([0, 1, 2, 3, 5, 7, 10, 25, 99])))
        K, C = rng.randint(0, 3), rng.randint(0, 3)
        sol = solve(n, edges, K, C)
        got = sol["cost"] if sol else -1
        want = brute_force(n, edges, K, C)
        # also verify the reported route is consistent
        if sol and sol["flights"]:
            ok = (sol["nodes"][0] == 1 and sol["nodes"][-1] == n
                  and len(sol["flights"]) <= K + 1
                  and sum(f["coupon"] for f in sol["flights"]) <= C
                  and sum(f["paid"] for f in sol["flights"]) == got)
            if not ok:
                bad += 1
                print("inconsistent route", n, edges, K, C, sol)
        if got != want:
            bad += 1
            if bad <= 5:
                print("MISMATCH", n, edges, K, C, "dp", got, "brute", want)
    print(f"Self-test: {trials} random graphs, {bad} problems")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--csv", help="option 2: read the flight graph from this CSV")
    src.add_argument("--generate", type=int, metavar="N", help="option 3: random graph with N cities")
    ap.add_argument("--csv-format", choices=["auto", "edges", "matrix"], default="auto")
    ap.add_argument("--n", type=int, help="number of cities (edge-list CSV only)")
    ap.add_argument("--edges", type=int, help="number of flights to generate (default 3N)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="flights_solution.csv")
    ap.add_argument("--input-out", default="flights_input.csv")
    ap.add_argument("--gif", default="travel_animation.gif")
    ap.add_argument("--png", default="travel_final.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if self_test() else 1)

    try:
        if args.csv:
            K, C = read_KC(sys.stdin.readline())
            n, edges = read_graph_csv(args.csv, args.csv_format, args.n)
            validate(n, edges, K, C)
        elif args.generate:
            K, C = read_KC(sys.stdin.readline())
            rng = random.Random(args.seed)
            n = args.generate
            edges = generate(n, args.edges or 3 * n, K, rng)
            validate(n, edges, K, C)
            write_edges_csv(args.input_out, edges)
            print(f"Generated graph saved to {args.input_out}", file=sys.stderr)
        else:
            n, edges, K, C = read_option1(sys.stdin.read())
    except ValueError as exc:
        sys.exit(f"Input error: {exc}")

    sol = solve(n, edges, K, C)
    print(sol["cost"] if sol else -1)
    if sol:
        parts = [str(sol["nodes"][0])]
        for fl in sol["flights"]:
            parts.append(f"-[{fl['price']}->{fl['paid']} coupon]->" if fl["coupon"]
                         else f"-[{fl['price']}]->")
            parts.append(str(fl["v"]))
        print("Route:", " ".join(parts))
        print(f"Flights: {len(sol['flights'])}, stops: {max(0, len(sol['flights']) - 1)}/{K}, "
              f"coupons used: {sum(f['coupon'] for f in sol['flights'])}/{C}")
    else:
        print(f"No route from city 1 to city {n} with at most {K} stops.")
    write_solution_csv(args.out, edges, sol)
    print(f"Saved {args.out}")
    if not args.no_viz:
        visualize(n, edges, sol, K, C, args.gif, args.png, args.show)


if __name__ == "__main__":
    main()
