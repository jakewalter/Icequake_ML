#!/usr/bin/env python3
"""Flat-layered 1D ray tracer for the vels1d reflection profile.

Replaces the straight-ray / single-average-velocity forward model that every S-P depth test
in this project has used so far. That approximation is fine at a station 0.2 km from the
epicentre and badly wrong at 5-6 km, and the failure is not subtle: fitting straight rays to
T2's CC-refined S-P over 0.21-6 km offsets returned per-station depths of 1.94 km (JULA,
r=0.21) and 0.00 km (WICH/EPJZ, r~4.9), and a joint fit that pushed clusters BELOW the bed
with Vp/Vs 1.80 against the model's 2.01. The far stations' rays bottom in faster material
than the average, arrive earlier than a straight ray predicts, and the fit absorbs the error
as fake shallow depth.

Physics: for a flat stack of homogeneous layers and ray parameter p (s/km), a layer of
thickness dz and velocity v contributes

    eta = sqrt(1/v^2 - p^2)          (vertical slowness; real only while p*v < 1)
    dx  = dz * p / eta               (horizontal distance)
    dt  = dz / (v^2 * eta)           (traveltime)

Two branches reach a surface receiver from a source at depth zs:

  * DIRECT   -- upgoing only, integrating 0..zs.
  * TURNING  -- down from zs to the depth where p*v = 1, then up over 0..z_turn. Because the
                ice-bed transition takes Vp from 3.85 to 5.66 km/s, this branch is the FIRST
                arrival at large offset, which is precisely what the straight-ray model has
                no way to represent.

The first arrival is the minimum over both branches. Traveltimes are tabulated on a (z, r)
grid once and bilinearly interpolated, so a depth inversion can call this in a loop.

Sign conventions and units: depth and distance in km, velocity km/s, time in s.
"""
import numpy as np


class Model1D:
    """A 1D velocity model sampled finely enough to treat as constant-velocity layers."""

    def __init__(self, depth_km, v_km_s):
        d = np.asarray(depth_km, float)
        v = np.asarray(v_km_s, float)
        ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
        d, v = d[ok], v[ok]
        order = np.argsort(d)
        self.z = d[order]
        self.v = v[order]
        # layer i spans [z[i], z[i+1]) with velocity v[i]
        self.dz = np.diff(self.z)
        self.vl = self.v[:-1]
        self.zt = self.z[:-1]

    def _legs(self, p):
        """Per-layer (dx, dt) for ray parameter p; NaN where the ray cannot propagate."""
        with np.errstate(invalid="ignore", divide="ignore"):
            eta = np.sqrt(1.0 / self.vl**2 - p * p)
            dx = self.dz * p / eta
            dt = self.dz / (self.vl**2 * eta)
        bad = ~np.isfinite(eta) | (eta <= 0)
        dx[bad] = np.nan
        dt[bad] = np.nan
        return dx, dt

    def _branch_direct(self, p, zs):
        """Upgoing ray, source depth zs to the surface."""
        n = np.searchsorted(self.zt, zs, side="right")
        if n == 0:
            return 0.0, 0.0
        dx, dt = self._legs(p)
        dx, dt = dx[:n].copy(), dt[:n].copy()
        # trim the partial bottom layer so the ray starts exactly at zs
        frac = (zs - self.zt[n - 1]) / self.dz[n - 1]
        dx[-1] *= frac
        dt[-1] *= frac
        if not np.all(np.isfinite(dx)):
            return np.nan, np.nan
        return float(dx.sum()), float(dt.sum())

    def _branch_turning(self, p, zs):
        """Down from zs to the turning depth (p*v = 1), then up the whole column."""
        dx, dt = self._legs(p)
        turn = np.nonzero(~np.isfinite(dx))[0]
        # first layer the ray cannot enter, searched BELOW the source
        n_src = np.searchsorted(self.zt, zs, side="right")
        turn = turn[turn >= n_src]
        if turn.size == 0:
            return np.nan, np.nan          # never turns inside the model
        k = turn[0]
        if k <= n_src:
            return np.nan, np.nan          # turns at or above the source: that is the direct ray
        # up-leg: surface to turning depth; down-leg: source to turning depth
        up_x, up_t = np.nansum(dx[:k]), np.nansum(dt[:k])
        dn_x, dn_t = np.nansum(dx[n_src:k]), np.nansum(dt[n_src:k])
        # correct the source's partial layer on the down-leg
        frac = 1.0 - (zs - self.zt[n_src - 1]) / self.dz[n_src - 1] if n_src > 0 else 1.0
        dn_x += dx[n_src - 1] * frac if n_src > 0 else 0.0
        dn_t += dt[n_src - 1] * frac if n_src > 0 else 0.0
        return float(up_x + dn_x), float(up_t + dn_t)

    def _branch_head(self, zs, r):
        """Critically refracted (head) wave along the top of the fastest material.

        A turning ray needs a velocity GRADIENT. Below ~2.6 km this profile is effectively a
        constant 5.66 km/s half-space, so no ray turns there and the far-offset first arrival
        is instead the head wave grazing its top at v_max. Without this branch the tracer
        silently returns the (slower) direct arrival past the crossover -- validated at 45 ms
        too slow at 7 km on a sharp two-layer test.
        """
        v_max = self.vl.max()
        p = 1.0 / v_max
        k = int(np.argmax(self.vl >= v_max - 1e-12))   # top of the fastest material
        n_src = np.searchsorted(self.zt, zs, side="right")
        if k <= n_src:
            return np.nan                              # source at or below the refractor
        dx, dt = self._legs(p * (1 - 1e-12))
        up_x, up_t = np.nansum(dx[:k]), np.nansum(dt[:k])
        dn_x, dn_t = np.nansum(dx[n_src:k]), np.nansum(dt[n_src:k])
        x_legs, t_legs = up_x + dn_x, up_t + dn_t
        if not np.isfinite(x_legs) or r < x_legs:
            return np.nan                              # inside the crossover: no head wave
        return float(t_legs + (r - x_legs) / v_max)

    def traveltime(self, zs, r, n_p=900):
        """First-arrival time from a source at depth `zs` to a surface receiver at offset `r`."""
        n_src = max(np.searchsorted(self.zt, zs, side="right"), 1)
        p_max = 1.0 / self.vl[:n_src].max()
        ps = np.linspace(0.0, p_max * (1 - 1e-9), n_p)

        best = np.inf
        for branch in (self._branch_direct, self._branch_turning):
            xs, ts = [], []
            for p in ps:
                x, t = branch(p, zs)
                if np.isfinite(x) and np.isfinite(t):
                    xs.append(x)
                    ts.append(t)
            if len(xs) < 2:
                continue
            xs, ts = np.asarray(xs), np.asarray(ts)
            o = np.argsort(xs)
            xs, ts = xs[o], ts[o]
            if r < xs[0] or r > xs[-1]:
                continue
            best = min(best, float(np.interp(r, xs, ts)))
        hw = self._branch_head(zs, r)
        if np.isfinite(hw):
            best = min(best, hw)
        return best if np.isfinite(best) else np.nan


class SPTable:
    """Tabulated S-P time on a (depth, offset) grid, bilinearly interpolated."""

    def __init__(self, depth_km, vp, vs, z_grid, r_grid):
        self.mp = Model1D(depth_km, vp)
        self.ms = Model1D(depth_km, vs)
        self.z_grid = np.asarray(z_grid, float)
        self.r_grid = np.asarray(r_grid, float)
        # The (x, t) curves depend only on the source depth, so sweep p ONCE per depth and
        # interpolate every offset off the same curve -- not once per (depth, offset) pair.
        self.tp = np.array([self._row(self.mp, z) for z in self.z_grid])
        self.ts = np.array([self._row(self.ms, z) for z in self.z_grid])
        self.sp = self.ts - self.tp

    def _row(self, model, z, n_p=900):
        n_src = max(np.searchsorted(model.zt, z, side="right"), 1)
        ps = np.linspace(0.0, (1.0 / model.vl[:n_src].max()) * (1 - 1e-9), n_p)
        out = np.full(len(self.r_grid), np.inf)
        for branch in (model._branch_direct, model._branch_turning):
            pts = [branch(p, z) for p in ps]
            pts = [(x, t) for x, t in pts if np.isfinite(x) and np.isfinite(t)]
            if len(pts) < 2:
                continue
            xs, ts = map(np.asarray, zip(*pts))
            o = np.argsort(xs)
            xs, ts = xs[o], ts[o]
            inside = (self.r_grid >= xs[0]) & (self.r_grid <= xs[-1])
            cand = np.where(inside, np.interp(self.r_grid, xs, ts), np.inf)
            out = np.minimum(out, cand)
        hw = np.array([model._branch_head(z, r) for r in self.r_grid])
        out = np.minimum(out, np.where(np.isfinite(hw), hw, np.inf))
        return np.where(np.isfinite(out), out, np.nan)

    def __call__(self, z, r):
        """S-P (seconds) at depth z, offset r; arrays allowed."""
        z = np.atleast_1d(np.asarray(z, float))
        r = np.atleast_1d(np.asarray(r, float))
        zi = np.clip(np.searchsorted(self.z_grid, z) - 1, 0, len(self.z_grid) - 2)
        ri = np.clip(np.searchsorted(self.r_grid, r) - 1, 0, len(self.r_grid) - 2)
        z0, z1 = self.z_grid[zi], self.z_grid[zi + 1]
        r0, r1 = self.r_grid[ri], self.r_grid[ri + 1]
        fz = np.clip((z - z0) / (z1 - z0), 0, 1)
        fr = np.clip((r - r0) / (r1 - r0), 0, 1)
        g = self.sp
        return ((1 - fz) * (1 - fr) * g[zi, ri] + (1 - fz) * fr * g[zi, ri + 1]
                + fz * (1 - fr) * g[zi + 1, ri] + fz * fr * g[zi + 1, ri + 1])


def t2_sp_table(z_grid=None, r_grid=None, site="T2"):
    """S-P table for a site's vels1d profile (thickness-shifted as vels1d_model defines)."""
    import vels1d_model
    depth_m, vp_ms, vs_ms = vels1d_model.shifted_profile(site)
    keep = depth_m <= 8000.0
    z_grid = np.arange(0.05, 3.51, 0.05) if z_grid is None else z_grid
    r_grid = np.arange(0.10, 7.01, 0.10) if r_grid is None else r_grid
    return SPTable(depth_m[keep] / 1000.0, vp_ms[keep] / 1000.0, vs_ms[keep] / 1000.0,
                   z_grid, r_grid)
