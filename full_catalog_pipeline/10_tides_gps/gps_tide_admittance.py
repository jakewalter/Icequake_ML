#!/usr/bin/env python3
"""Fit a tide -> GPS-velocity transfer function, then predict velocity where GPS is absent.

The GPS record (cache_disp_v3, the cleaned/detrended product) covers 2020-01-02 .. 2020-04-22
-- about 100 days of the seismic catalogue's 710. So: fit a complex ADMITTANCE per tidal
constituent on the overlap, then synthesise velocity over the whole catalogue from the tide
alone.

Why an admittance rather than a regression on tide height. The ice-stream response is
frequency dependent and lagged: a 140-209 km inland site cannot respond in phase with the
grounding-zone tide. Fitting one gain against tide height (which is what a scalar regression
does) forces a single lag on all constituents and is why the earlier zero-lag correlation came
out at |r| ~ 0.01. Instead, harmonically analyse BOTH series over the common window and take

    Z_i = H_gps,i / H_tide,i        (complex, per constituent i)

which carries its own gain AND phase per frequency. Prediction is then
sum_i Re{Z_i * H_tide,i * exp(i w_i t)}, valid at any time the tide can be synthesised.

TIDE SOURCE. CATS2008_v2023, synthesised with pyTMD's own OTIS-convention predictor.
Round-trip validated: re-fitting the synthesised series recovers the input constituent
amplitudes to 0.99-1.12 (deviations are nodal, largest for K1/O1/Mf as expected) with variance
explained 0.99937.

DO NOT USE gps_data/thwaites_tidal_predictions.csv. That file is estimate_tides_pytmd.py's
FALLBACK, not a model: four hardcoded amplitudes (M2 0.50, S2 0.15, K1 0.25, O1 0.20 m) with
every other constituent exactly zero. It is M2-dominant where the real local tide is
diurnal-dominant (CATS2008: M2 5 cm vs K1 34 cm), so any constituent-level or phase conclusion
drawn from it is unreliable.

THE ARRAYS ARE OUTSIDE THE TIDE MODEL. Both sit on mask=0; the nearest wet cell is 209 km
(T1) and 140 km (T2) away. The tide used is therefore the grounding-zone forcing, and the
admittance absorbs the entire unknown ice-stream transfer between there and the site.

Usage:
    python full_catalog_pipeline/gps_tide_admittance.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import pyTMD.predict as PP

GPS_DIR = "gps_data/cache_disp_v3"
CATS = "CATS2008_v2023.nc"
CON = "m2 s2 n2 k2 k1 o1 p1 q1 mf mm".split()
PERIODS = np.array([12.420601, 12.0, 12.658348, 11.967235, 23.934470,
                    25.819342, 24.065890, 26.868357, 327.859, 661.31])
# from offset_velocity_v2.py
STATIONS = {
    "T02A": (-76.48207, -103.05797), "T02B": (-76.51003, -103.42821),
    "T02C": (-76.52372, -103.61386), "T01A": (-77.25144, -100.02871),
    "T01B": (-77.33347, -100.20739), "T01C": (-77.37449, -100.29784),
    "T01D": (-77.41409, -100.38941),
}
C1, C2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def nearest_wet(ds, lat0, lon0):
    lat, lon, mask = ds["lat"][:], ds["lon"][:], ds["mask"][:]
    d = np.hypot((lat - lat0) * 111.0, (lon - lon0) * 111.0 * np.cos(np.radians(lat0)))
    dw = np.where(mask > 0, d, np.inf)
    k = np.unravel_index(np.argmin(dw), dw.shape)
    return k, float(dw[k]), float(lat[k]), float(lon[k])


def tide_at(ds, k, times):
    hRe = np.asarray(ds["hRe"][:, k[0], k[1]], float)
    hIm = np.asarray(ds["hIm"][:, k[0], k[1]], float)
    hc = np.ma.masked_array((hRe + 1j * hIm)[None, :], mask=np.zeros((1, len(CON)), bool))
    td = (times - pd.Timestamp("1992-01-01")).total_seconds().values / 86400.0
    return np.asarray(PP.time_series(td, hc, CON, corrections="OTIS")).ravel()


def design(t_hours):
    """sin/cos design matrix for the 10 constituents (plus a constant)."""
    cols = [np.ones_like(t_hours)]
    for P in PERIODS:
        cols += [np.sin(2 * np.pi * t_hours / P), np.cos(2 * np.pi * t_hours / P)]
    return np.column_stack(cols)


def harmonic_fit(t_hours, y, w=None):
    """Complex amplitude per constituent: H_i = a_i - i b_i for y ~ a sin + b cos."""
    A = design(t_hours)
    if w is None:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    else:
        beta, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return np.array([beta[1 + 2 * i] - 1j * beta[2 + 2 * i] for i in range(len(PERIODS))]), \
        A, beta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",
                    default="full_catalog_pipeline/artifacts/full_run/T2_v5")
    ap.add_argument("--field", default="vel_sm",
                    help="GPS column to model (vel_sm = smoothed velocity)")
    args = ap.parse_args()

    ds = nc.Dataset(CATS)
    rows, preds = [], {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for ai, arr in enumerate(("T01", "T02")):
        stations = [s for s in STATIONS if s.startswith(arr)]
        la, lo = np.mean([STATIONS[s][0] for s in stations]), \
                 np.mean([STATIONS[s][1] for s in stations])
        k, dkm, wla, wlo = nearest_wet(ds, la, lo)
        print("=" * 76)
        print(f"{arr}: {len(stations)} stations; tide from CATS cell "
              f"({wla:.3f}, {wlo:.3f}), {dkm:.0f} km away")
        print("=" * 76)

        for s in sorted(stations):
            g = pd.read_parquet(f"{GPS_DIR}/{s}.parquet")
            y = g[args.field]
            ok = y.notna().values
            if ok.sum() < 5000:
                print(f"  {s}: only {ok.sum()} valid samples, skipped")
                continue
            times = g.index[ok].tz_localize(None)
            z = tide_at(ds, k, times)
            t_h = (times - times[0]).total_seconds().values / 3600.0
            Hg, _, _ = harmonic_fit(t_h, y.values[ok])
            Hz, _, _ = harmonic_fit(t_h, z)
            Z = Hg / Hz                      # complex admittance
            # response significance: variance of the GPS series explained by the tidal fit
            A = design(t_h)
            beta, *_ = np.linalg.lstsq(A, y.values[ok], rcond=None)
            r2 = 1 - np.var(y.values[ok] - A @ beta) / np.var(y.values[ok])
            print(f"  {s}: n={ok.sum():6d}  tidal fit explains {r2:6.2%} of {args.field}")
            for i, c in enumerate(CON):
                rows.append(dict(array=arr, station=s, constituent=c.upper(),
                                 period_h=PERIODS[i],
                                 tide_amp_cm=abs(Hz[i]) * 100,
                                 gps_amp=abs(Hg[i]),
                                 admittance_gain=abs(Z[i]),
                                 admittance_phase_deg=np.degrees(np.angle(Z[i])),
                                 r2_total=r2, n=int(ok.sum())))

        # ---- prediction over the whole catalogue, from the tide alone ----
        full = pd.date_range("2020-01-11", "2021-12-21", freq="1h")
        zf = tide_at(ds, k, full)
        # use the array-median admittance so one bad station cannot dominate
        sub = pd.DataFrame([r for r in rows if r["array"] == arr])
        gains = sub.groupby("constituent").admittance_gain.median()
        phases = sub.groupby("constituent").admittance_phase_deg.median()
        tf = (full - full[0]).total_seconds().values / 3600.0
        Hzf, _, _ = harmonic_fit(tf, zf)
        vpred = np.zeros(len(full))
        for i, c in enumerate(CON):
            g_, p_ = gains[c.upper()], np.radians(phases[c.upper()])
            amp = abs(Hzf[i]) * g_
            ph = np.angle(Hzf[i]) + p_
            vpred += amp * np.sin(2 * np.pi * tf / PERIODS[i] + ph)
        preds[arr] = pd.DataFrame(dict(datetime=full, tide_m=zf, vel_pred=vpred))
        print(f"  predicted {args.field} over {len(full)} hourly steps: "
              f"std {vpred.std():.4f}, range {vpred.min():.3f}..{vpred.max():.3f}")

        # figures
        ax = axes[0, ai]
        sub2 = sub[sub.constituent.isin(["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"])]
        for j, s in enumerate(sorted(sub2.station.unique())):
            d = sub2[sub2.station == s]
            ax.plot(range(len(d)), d.admittance_gain, "o-", ms=5, lw=1.2,
                    label=s, alpha=0.85)
        ax.set_xticks(range(8))
        ax.set_xticklabels(["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"], fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel(f"admittance gain\n({args.field} per m of tide)", fontsize=9.5,
                      color=MUTED)
        ax.set_title(f"{arr}: tide -> GPS admittance, per constituent",
                     fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)

        ax = axes[1, ai]
        m = (preds[arr].datetime >= "2020-02-01") & (preds[arr].datetime < "2020-03-01")
        ax.plot(preds[arr].datetime[m], preds[arr].vel_pred[m],
                color=(C1 if arr == "T01" else C2), lw=1.4, zorder=3,
                label="predicted from tide")
        s0 = sorted(stations)[0]
        g = pd.read_parquet(f"{GPS_DIR}/{s0}.parquet")
        gm = (g.index.tz_localize(None) >= pd.Timestamp("2020-02-01")) & \
             (g.index.tz_localize(None) < pd.Timestamp("2020-03-01"))
        obs = g[args.field][gm]
        ax.plot(g.index[gm].tz_localize(None), obs - np.nanmean(obs), color=MUTED, lw=0.9,
                alpha=0.8, zorder=2, label=f"observed {s0} (mean removed)")
        ax.set_ylabel(args.field, fontsize=9.5, color=MUTED)
        ax.set_title(f"{arr}: February 2020 — prediction vs observation",
                     fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=8, frameon=False)
        print()

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "gps_tide_admittance.csv"), index=False)
    for arr, p in preds.items():
        p.to_csv(os.path.join(args.out_dir, f"{arr}_predicted_velocity.csv"), index=False)

    fig.suptitle("Tide -> GPS velocity admittance (CATS2008_v2023) and "
                 "prediction beyond the GPS record", fontsize=13, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(args.out_dir, "gps_tide_admittance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {out}")
    print(f"wrote {os.path.join(args.out_dir, 'gps_tide_admittance.csv')}")
    for arr in preds:
        print(f"wrote {os.path.join(args.out_dir, f'{arr}_predicted_velocity.csv')}")


if __name__ == "__main__":
    main()
