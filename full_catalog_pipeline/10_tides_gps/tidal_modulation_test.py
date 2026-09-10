#!/usr/bin/env python3
"""Is T1/T2 icequake timing modulated at tidal periods?

Two independent tests, neither of which needs an ocean tide model grid:

  A. SPECTRAL. Power spectrum of the hourly event-rate series, with the diurnal and
     semidiurnal tidal periods marked. 710 days of record resolves M2 (12.4206 h) from
     S2 (12.0000 h) comfortably -- their beat period is 14.77 d, so ~48 cycles are covered.

  B. PHASE (Schuster test). For each event, the astronomical phase of a given constituent
     at that instant, from pyTMD's Delaunay/Doodson arguments. If events are unmodulated the
     phases are uniform on [0, 2pi); the Schuster statistic tests that. This is the standard
     tidal-triggering test and uses ONLY astronomy.

SCOPE LIMIT. The within-day shuffle preserves each day's event COUNT exactly, so the DAILY
series is identical between observed and null. The spectral test therefore has essentially no
discriminating power at periods much longer than a day: Mf/Msf/Mm are NOT testable this way
and are excluded. An earlier version reported them as highly significant; that was artifact.

SOLAR CONFOUND. A 24.000 h cycle is present at both arrays (time-of-day rate varies 45-59%
peak-to-trough). P1 (24.0659 h) and K1 (23.9345 h) lie within ~2 Rayleigh widths of it, so
solar power leaks into both. All tidal claims are made AFTER regressing out 24/12/8/6 h.

NO ANALYTIC SCHUSTER p. exp(-R^2/n) assumes independent events; these arrive in swarms. For
Mf it returned 1.4e-38 against a shuffle-null p of 0.28 -- inflation ~1e37. Shuffle only.

THE NULL IS THE WHOLE BALLGAME. These catalogues are violently bursty -- a swarm dumps
hundreds of events into a day -- and burstiness alone produces broadband spectral power and
apparent phase clustering. So the null preserves each day's event COUNT exactly and
redistributes the times uniformly within that day. That destroys sub-daily periodicity while
leaving the multi-day burst envelope untouched, which is precisely the comparison wanted.
A white-noise or Poisson null would be far too permissive and would manufacture significance.

WHAT THE PHASE TEST CANNOT TELL YOU. The astronomical phase is not the ocean tide phase at
this location: the ocean response carries a site-specific amplitude and lag, and inland of a
grounding line the stress is transmitted by flexure with further lag. So a significant
Schuster result establishes MODULATION AT THAT PERIOD, but the preferred phase must not be
read as "events occur at high tide" without a real tide model (CATS2008/TPXO) for the site.

Usage:
    python full_catalog_pipeline/tidal_modulation_test.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

# hours; the classical constituents that matter for ice-shelf / grounding-line forcing
CONSTITUENTS = {
    "M2": 12.420601, "S2": 12.000000, "N2": 12.658348, "K2": 11.967235,
    "K1": 23.934470, "O1": 25.819342, "P1": 24.065890, "Q1": 26.868357,
    "Mf": 13.660791 * 24, "Msf": 14.765294 * 24, "Mm": 27.554550 * 24,
}
SEMIDIURNAL = ["M2", "S2", "N2", "K2"]
DIURNAL = ["K1", "O1", "P1", "Q1"]

C1, C2 = "#2a78d6", "#eb6834"
C_SIG = "#008300"   # green: constituent significant on the spectral test
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"
N_NULL = 500


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def load_times(path):
    d = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    t = (pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hr, minute=d.mi))
         + pd.to_timedelta(d.sc.astype(float), unit="s"))
    return t.sort_values().reset_index(drop=True)


def hourly_series(t, t0, t1):
    hours = pd.date_range(t0.floor("D"), t1.ceil("D"), freq="h")
    s = pd.Series(1, index=t).resample("h").sum().reindex(hours, fill_value=0)
    return hours, s.values.astype(float)


def remove_solar(hours, x):
    """Regress out the SOLAR lines (24, 12, 8, 6 h) at exactly those periods.

    A 24.000 h thermal/solar cycle is present at both arrays (time-of-day rate varies 45-59%
    peak-to-trough) and P1 (24.0659 h) and K1 (23.9345 h) sit within ~2 Rayleigh widths of
    it, so solar power leaks into both. Every tidal claim must be made on the residual.
    """
    t = (hours - hours[0]).total_seconds().values / 3600.0
    cols = [np.ones_like(t)]
    for P in (24.0, 12.0, 8.0, 6.0):
        cols += [np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def spectrum(x, dt_hours=1.0):
    """Amplitude spectrum of a mean-removed series; returns (period_hours, power)."""
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, d=dt_hours)          # cycles per hour
    p = np.abs(X) ** 2
    with np.errstate(divide="ignore"):
        per = 1.0 / f
    return per[1:], p[1:]


def within_day_shuffle(t, rng):
    """Keep each day's event count; randomize the time-of-day. Preserves burst envelope."""
    day = t.dt.floor("D")
    frac = rng.random(len(t))
    return day + pd.to_timedelta(frac * 24.0, unit="h")


def schuster(phases):
    """Schuster test. Returns (R/n vector strength, p-value, preferred phase in degrees)."""
    n = len(phases)
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R2 = C * C + S * S
    p = float(np.exp(-R2 / n))                  # Schuster's p
    return float(np.sqrt(R2) / n), p, float(np.degrees(np.arctan2(S, C)) % 360)


def constituent_phase(t, period_hours):
    """Phase of a constituent at times t, from its period and a fixed epoch.

    Deliberately NOT pyTMD's nodal-corrected argument: for a UNIFORMITY test only the
    steadily-advancing part matters, and using a pure period keeps the null exactly uniform.
    pyTMD is used separately to confirm the constituent frequencies below.
    """
    epoch = pd.Timestamp("2020-01-01")
    hrs = (t - epoch).dt.total_seconds().values / 3600.0
    return (2 * np.pi * hrs / period_hours) % (2 * np.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    R = "full_catalog_pipeline/artifacts/full_run"
    ap.add_argument("--t1", default=f"{R}/T1_v5/hypodd_vels1d_t1ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--t2", default=f"{R}/T2_v5/hypodd_vels1d_ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--out-dir", default=f"{R}/T2_v5")
    args = ap.parse_args()

    # cross-check our constituent periods against pyTMD's own frequencies
    try:
        import pyTMD.arguments as PA
        omega = PA.frequency(["m2", "s2", "n2", "k1", "o1", "p1"])
        pytmd_per = 2 * np.pi / np.array(omega) / 3600.0
        print("pyTMD frequency cross-check (period, hours):")
        for c, p in zip(["M2", "S2", "N2", "K1", "O1", "P1"], pytmd_per):
            print(f"   {c}: pyTMD {p:10.6f}   table {CONSTITUENTS[c]:10.6f}   "
                  f"diff {abs(p-CONSTITUENTS[c])*3600:.2f} s")
    except Exception as exc:
        print(f"pyTMD cross-check unavailable ({exc}); using tabulated periods")
    print()

    t1, t2 = load_times(args.t1), load_times(args.t2)
    lo = max(t1.min(), t2.min())
    hi = min(t1.max(), t2.max())
    t1 = t1[(t1 >= lo) & (t1 <= hi)].reset_index(drop=True)
    t2 = t2[(t2 >= lo) & (t2 <= hi)].reset_index(drop=True)
    rng = np.random.default_rng(0)

    SUBDAILY = SEMIDIURNAL + DIURNAL
    results = {}
    for name, t in (("T1", t1), ("T2", t2)):
        print("=" * 74)
        print(f"{name}: {len(t)} events, {lo.date()} .. {hi.date()}")
        print("=" * 74)
        hours, x_raw = hourly_series(t, lo, hi)
        x = remove_solar(hours, x_raw)
        per, pw = spectrum(x)

        # null: keep each day's count, randomize time-of-day, apply the SAME solar removal
        null_pw = np.empty((N_NULL, len(pw)))
        null_R = {c: np.empty(N_NULL) for c in SUBDAILY}
        for i in range(N_NULL):
            ts = within_day_shuffle(t, rng).sort_values().reset_index(drop=True)
            _, xs_raw = hourly_series(ts, lo, hi)
            null_pw[i] = spectrum(remove_solar(hours, xs_raw))[1]
            for c in SUBDAILY:
                null_R[c][i] = schuster(constituent_phase(ts, CONSTITUENTS[c]))[0]

        print(f"{'constituent':>11s} {'period(h)':>10s} {'spec p':>8s} "
              f"{'vector R':>9s} {'phase p':>8s}   verdict")
        rows = []
        for c in SUBDAILY:
            ph = CONSTITUENTS[c]
            j = int(np.argmin(np.abs(per - ph)))
            spec_p = float((null_pw[:, j] >= pw[j]).mean())
            Rv, _, pref = schuster(constituent_phase(t, ph))
            phase_p = float((null_R[c] >= Rv).mean())
            both = spec_p < 0.05 and phase_p < 0.05
            verdict = "BOTH tests" if both else (
                "one test only" if min(spec_p, phase_p) < 0.05 else "-")
            rows.append(dict(constituent=c, period_h=ph, spec_p=spec_p, vector_R=Rv,
                             phase_p=phase_p, preferred_phase_deg=pref, verdict=verdict))
            print(f"{c:>11s} {ph:10.4f} {spec_p:8.3f} {Rv:9.4f} {phase_p:8.3f}   {verdict}")
        results[name] = dict(per=per, pw=pw, null=null_pw, rows=pd.DataFrame(rows),
                             times=t, hours=hours, x=x, x_raw=x_raw)
        print()

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, name in enumerate(("T1", "T2")):
        r = results[name]
        c = C1 if name == "T1" else C2
        ax = axes[0, col]
        m = (r["per"] >= 10) & (r["per"] <= 30)
        ax.semilogy(r["per"][m], r["pw"][m], color=c, lw=1.0, zorder=3, label="observed")
        ax.semilogy(r["per"][m], np.percentile(r["null"][:, m], 95, axis=0), color=MUTED,
                    lw=1.2, ls="--", zorder=4, label="null p95 (within-day shuffle)")
        rr = r["rows"].set_index("constituent")
        for k, cn in enumerate(SEMIDIURNAL + DIURNAL):
            sig = rr.loc[cn, "spec_p"] < 0.05
            ax.axvline(CONSTITUENTS[cn], color=(C_SIG if sig else MUTED), lw=1.4 if sig else 0.7,
                       alpha=0.9 if sig else 0.35, zorder=2)
            ax.annotate(cn, (CONSTITUENTS[cn], 1.005 + 0.055 * (k % 2)), xycoords=("data", "axes fraction"),
                        fontsize=7.5, color=(INK if sig else MUTED),
                        fontweight="bold" if sig else "normal", ha="center", va="bottom")
        ax.set_xlabel("period (hours)", fontsize=9.5, color=MUTED)
        ax.set_ylabel("spectral power", fontsize=9.5, color=MUTED)
        ax.set_title(f"{name}: tidal-band spectrum of hourly event rate",
                     fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="lower right")

        # phase histograms: the strongest diurnal detection, and M2 as the null case
        ax = axes[1, col]
        rr = r["rows"].set_index("constituent")
        diur = rr.loc[[c for c in DIURNAL]].sort_values("phase_p")
        best = diur.index[0]
        for cn, colr, alpha in ((best, c, 0.85), ("M2", MUTED, 0.45)):
            ph = np.degrees(constituent_phase(r["times"], CONSTITUENTS[cn]))
            row = rr.loc[cn]
            ax.hist(ph, bins=24, range=(0, 360), color=colr, alpha=alpha, zorder=3,
                    label=f"{cn}: R={row.vector_R:.4f}, spec p={row.spec_p:.3f}, "
                          f"phase p={row.phase_p:.3f}")
        ax.axhline(len(r["times"]) / 24, color=INK, ls="--", lw=1.2, zorder=5,
                   label="uniform expectation")
        ax.set_xlabel("astronomical phase (deg)", fontsize=9.5, color=MUTED)
        ax.set_ylabel("events", fontsize=9.5, color=MUTED)
        ax.set_title(f"{name}: strongest diurnal ({best}) vs M2 (null case)",
                     fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="lower center")

    fig.suptitle("Tidal modulation of icequake timing — solar cycle removed, burst-preserving null",
                 fontsize=13.5, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(args.out_dir, "tidal_modulation_test.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")

    allr = pd.concat([results[k]["rows"].assign(array=k) for k in results])
    csv = os.path.join(args.out_dir, "tidal_modulation_test.csv")
    allr.to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
