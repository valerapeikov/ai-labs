import numpy as np

# ---------- ПАРАМЕТРЫ (всё меняется здесь) ----------
ENV = dict(size=100.0, boundary="wrap")      # "wrap" (тор) или "reflect" (отражение)
AGENTS = dict(n=60, v_max=2.0, a_max=0.3)    # число птиц, макс. скорость и ускорение
RULES = dict(radius=15.0, sep_radius=4.0,    # радиус взаимодействия и радиус разделения
             w_c=0.01, w_a=0.05, w_s=0.1)    # веса: центрирование, выравнивание, разделение


def limit(v, max_len):
    """Обрезает длину каждого вектора (строки массива) до max_len."""
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, max_len / np.maximum(length, 1e-9))


class Flock:
    def __init__(self, env, agents, rules, seed=0):
        self.env, self.ag, self.rules = env, agents, rules
        rng = np.random.default_rng(seed)
        n = agents["n"]
        self.pos = rng.uniform(0, env["size"], (n, 2))             # случайные позиции
        ang = rng.uniform(0, 2 * np.pi, n)                          # случайные направления
        self.vel = (np.column_stack([np.cos(ang), np.sin(ang)])
                    * rng.uniform(0.5, 1, (n, 1)) * agents["v_max"])

    def step(self, dt=1.0):
        size, r = self.env["size"], self.rules
        n = len(self.pos)

        # delta[i, j] = pos[j] - pos[i]: векторы от каждой птицы ко всем остальным
        delta = self.pos[None, :, :] - self.pos[:, None, :]
        if self.env["boundary"] == "wrap":
            delta -= size * np.round(delta / size)    # кратчайший путь через границу тора
        dist = np.linalg.norm(delta, axis=2)

        eye = np.eye(n, dtype=bool)
        neigh = (dist < r["radius"]) & ~eye           # маска соседей (себя исключаем)
        cnt = np.maximum(neigh.sum(1), 1)[:, None]    # число соседей (минимум 1: защита от деления на 0)
        has = neigh.any(1)[:, None]                   # есть ли соседи вообще

        coh = (neigh[:, :, None] * delta).sum(1) / cnt     # средний delta = (центр соседей - моя позиция)
        ali = ((neigh @ self.vel) / cnt - self.vel) * has  # средняя скорость соседей - моя; 0 если соседей нет
        close = (dist < r["sep_radius"]) & ~eye
        sep = -(close[:, :, None] * delta).sum(1)          # сумма векторов "от" слишком близких

        acc = r["w_c"] * coh + r["w_a"] * ali + r["w_s"] * sep   # взвешенная сумма правил
        acc = limit(acc, self.ag["a_max"])                        # ограничение ускорения
        self.vel = limit(self.vel + acc * dt, self.ag["v_max"])   # обновление и ограничение скорости
        self.pos += self.vel * dt                                 # движение

        if self.env["boundary"] == "wrap":
            self.pos %= size                                      # вылетел справа - появился слева
        else:
            lo, hi = self.pos < 0, self.pos > size
            self.pos[lo] *= -1                                    # зеркально вернуть в поле
            self.pos[hi] = 2 * size - self.pos[hi]
            self.vel[lo | hi] *= -1                               # развернуть нужную компоненту скорости

    def metrics(self):
        """order: 1 - все летят одинаково, 0 - хаос; nn - среднее расстояние до ближайшего;
        coll - число пар ближе 0.5 (считаем столкновением)."""
        speed = np.linalg.norm(self.vel, axis=1).mean()
        order = np.linalg.norm(self.vel.mean(0)) / max(speed, 1e-9)
        d = self.pos[None] - self.pos[:, None]
        if self.env["boundary"] == "wrap":
            d -= self.env["size"] * np.round(d / self.env["size"])
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        return order, dist.min(1).mean(), (dist < 0.5).sum() // 2


def run(steps=600, env=None, agents=None, rules=None, seed=0):
    f = Flock({**ENV, **(env or {})}, {**AGENTS, **(agents or {})}, {**RULES, **(rules or {})}, seed)
    for _ in range(steps):
        f.step()
    return np.array(f.metrics(), dtype=float)


def avg_run(seeds=5, **kw):
    """Среднее по нескольким случайным стартам - чтобы выводы не зависели от одного запуска."""
    return np.mean([run(seed=s, **kw) for s in range(seeds)], axis=0)


# ---------- ЭКСПЕРИМЕНТЫ ----------
def experiments():
    base = dict(w_c=0.01, w_a=0.05, w_s=0.1)
    exps = [
        ("R=5",  dict(radius=5,  **base)),
        ("R=15", dict(radius=15, **base)),
        ("R=40", dict(radius=40, **base)),
        ("только центрирование",  dict(radius=15, w_c=0.01, w_a=0, w_s=0)),
        ("только выравнивание",   dict(radius=15, w_c=0, w_a=0.05, w_s=0)),
        ("только разделение",     dict(radius=15, w_c=0, w_a=0, w_s=0.1)),
        ("без разделения",        dict(radius=15, w_c=0.01, w_a=0.05, w_s=0)),
        ("сильное разделение",    dict(radius=15, w_c=0.01, w_a=0.05, w_s=0.5)),
        ("сильное центрирование", dict(radius=15, w_c=0.1, w_a=0.05, w_s=0.1)),
    ]
    print(f"{'эксперимент':26s} order  ср.до_ближ  столкновений")
    for name, rl in exps:
        o, nn, c = avg_run(rules=rl)
        print(f"{name:26s} {o:5.2f}  {nn:9.2f}  {c:8.1f}")
    for b in ("wrap", "reflect"):
        o, nn, c = avg_run(env=dict(boundary=b))
        print(f"границы {b:18s} {o:5.2f}  {nn:9.2f}  {c:8.1f}")


# ---------- АНИМАЦИЯ СО СЛАЙДЕРАМИ ----------
def animate():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider

    f = Flock(ENV, AGENTS, RULES)
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.32)
    ax.set_xlim(0, ENV["size"]); ax.set_ylim(0, ENV["size"]); ax.set_aspect("equal")
    q = ax.quiver(f.pos[:, 0], f.pos[:, 1], f.vel[:, 0], f.vel[:, 1],
                  pivot="mid", scale=40, color="tab:blue")   # стрелки = направление полёта

    def update(_):
        f.step()
        q.set_offsets(f.pos)
        q.set_UVC(f.vel[:, 0], f.vel[:, 1])
        return (q,)

    sliders = []   # храним ссылки, иначе слайдеры собирает сборщик мусора
    specs = [("radius", "Радиус", 1, 50), ("w_c", "Центрир.", 0, 0.1),
             ("w_a", "Выравн.", 0, 0.2), ("w_s", "Разделен.", 0, 0.5)]
    for i, (key, label, lo, hi) in enumerate(specs):
        s = Slider(fig.add_axes([0.2, 0.20 - i * 0.045, 0.65, 0.03]), label, lo, hi, valinit=RULES[key])
        s.on_changed(lambda v, k=key: f.rules.__setitem__(k, v))   # меняем правило на лету
        sliders.append(s)

    anim = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
    plt.show()
    return anim, sliders


if __name__ == "__main__":
    import sys
    experiments() if "--exp" in sys.argv else animate()