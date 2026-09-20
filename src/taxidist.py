"""Stand-to-runway taxi distance and turning, from OpenStreetMap aeroway geometry.

The model sees ADEP_STAND_RUNWAY as a 12,038-level categorical plus a smoothed
target encoding, so a rare pair falls back to the prior and the model has no idea
that stand A is 400 m from runway X and 3 km from runway Y. Distance is the
dominant physical term in taxi time; this makes it an explicit number that
generalises across pairs instead of being learned per level.

Turning angles matter alongside path length (Ravizza et al.): a route of the
same length with more turning is slower, so total turning and the detour ratio
are emitted separately rather than folded into the distance.

Geometry comes from Overpass (`dataset/geo/osm/<ICAO>_geom.json`, git-ignored;
refetch with `python src/taxidist.py fetch`). Ways carry both OSM node ids and
coordinates, so the graph is built on shared node ids and is topologically
correct rather than snapped by proximity.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

OSM_DIR = pathlib.Path("dataset/geo/osm")
TABLE_PATH = pathlib.Path("dataset/geo/taxi_dist.parquet")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
R_EARTH_M = 6371008.8


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _designators(ref):
    """'09L/27R' -> ['09L', '27R']; helipads and junk refs give []."""
    if not ref:
        return []
    parts = [p.strip().upper() for p in str(ref).split("/")]
    return [p for p in parts if p[:2].isdigit()]


class Airport:
    """Taxiway graph for one airport, with stands and runway thresholds located."""

    def __init__(self, apt):
        self.apt = apt
        els = json.loads((OSM_DIR / f"{apt}_geom.json").read_bytes())["elements"]
        self.coord = {}                       # osm node id -> (lat, lon)
        edges = []                            # (node id, node id)
        self.stands = {}                      # stand ref -> osm node id
        self.thresholds = {}                  # runway designator -> osm node id

        for e in els:
            tags = e.get("tags", {})
            way = tags.get("aeroway")
            if e["type"] == "way" and "geometry" in e and "nodes" in e:
                ids, geom = e["nodes"], e["geometry"]
                if len(ids) != len(geom):
                    continue
                for i, g in zip(ids, geom):
                    self.coord[i] = (g["lat"], g["lon"])
                if way in ("taxiway", "runway"):
                    edges.extend(zip(ids[:-1], ids[1:]))
                if way == "runway":
                    self._add_thresholds(tags.get("ref"), ids, geom)
                if way == "parking_position" and tags.get("ref"):
                    # a stand drawn as a stub way: its outer end is where the
                    # aircraft stands, but either end lands on the same apron
                    self.stands.setdefault(str(tags["ref"]), ids[0])
            elif e["type"] == "node" and way == "parking_position" and tags.get("ref"):
                self.coord[e["id"]] = (e["lat"], e["lon"])
                self.stands.setdefault(str(tags["ref"]), e["id"])

        self.nodes = sorted(self.coord)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.lat = np.array([self.coord[n][0] for n in self.nodes])
        self.lon = np.array([self.coord[n][1] for n in self.nodes])
        self._build_graph(edges)

    def _add_thresholds(self, ref, ids, geom):
        """Map each designator to the end you start the take-off roll from.

        Runway 14 points at ~140 degrees, so its threshold is the endpoint from
        which the bearing to the far end is ~140.
        """
        des = _designators(ref)
        if not des or len(des) > 2:
            return
        # EDDF's runway 18 is departure-only and mapped with a single designator,
        # which is 61% of EDDF departures -- do not require a reciprocal pair
        a, b = geom[0], geom[-1]
        brg = _bearing(a["lat"], a["lon"], b["lat"], b["lon"])
        for d in des:
            want = int(d[:2]) * 10.0
            diff = abs((brg - want + 180.0) % 360.0 - 180.0)
            self.thresholds[d] = ids[0] if diff < 90 else ids[-1]

    def _build_graph(self, edges):
        """Undirected weighted graph, plus a short connector for detached stands.

        Stands drawn as standalone nodes are not part of any taxiway way, so they
        would be unreachable. Each is joined to its nearest graph node; the
        connector length is real distance, so it is not free.
        """
        rows, cols, w = [], [], []
        for u, v in edges:
            iu, iv = self.index[u], self.index[v]
            d = float(_haversine(self.lat[iu], self.lon[iu], self.lat[iv], self.lon[iv]))
            rows.append(iu); cols.append(iv); w.append(d)

        on_way = set()
        for u, v in edges:
            on_way.add(u); on_way.add(v)
        way_idx = np.array([self.index[n] for n in sorted(on_way)]) if on_way else np.array([], int)
        for ref, node in self.stands.items():
            if node in on_way or way_idx.size == 0:
                continue
            i = self.index[node]
            d = _haversine(self.lat[i], self.lon[i], self.lat[way_idx], self.lon[way_idx])
            j = int(way_idx[np.argmin(d)])
            rows.append(i); cols.append(j); w.append(float(d.min()))

        n = len(self.nodes)
        self.graph = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()

    def routes(self):
        """One row per (stand, runway) pair: distance, total turning, detour ratio.

        The count of turns above a threshold is NOT emitted: it depends on how
        finely each mapper drew the taxiway centreline, so it is not comparable
        across airports. Measured against taxi time it is worthless (row-level
        correlation 0.009) while integrated turning, which is invariant to
        polyline granularity, reaches 0.178.
        """
        out = []
        for des, tnode in self.thresholds.items():
            src = self.index[tnode]
            dist, pred = dijkstra(self.graph, directed=False, indices=src,
                                  return_predecessors=True)
            for ref, snode in self.stands.items():
                j = self.index[snode]
                if not np.isfinite(dist[j]):
                    continue
                path = self._path(pred, src, j)
                total = self._turning(path)
                crow = float(_haversine(self.lat[src], self.lon[src],
                                        self.lat[j], self.lon[j]))
                detour = float(dist[j] / crow) if crow > 1.0 else np.nan
                out.append((self.apt, ref, des, float(dist[j]), total, detour))
        return pd.DataFrame(out, columns=["ADEP_mvt", "stand", "runway", "osm_taxi_dist_m",
                                          "osm_taxi_turn_deg", "osm_taxi_detour"])

    @staticmethod
    def _path(pred, src, dst):
        path, cur = [], dst
        while cur != src and cur >= 0:
            path.append(cur)
            cur = pred[cur]
        path.append(src)
        return path[::-1]

    def _turning(self, path):
        """Total absolute heading change along a path, in degrees."""
        if len(path) < 3:
            return 0.0
        la, lo = self.lat[path], self.lon[path]
        brg = _bearing(la[:-1], lo[:-1], la[1:], lo[1:])
        return float(np.abs((np.diff(brg) + 180.0) % 360.0 - 180.0).sum())


def build_table(verbose=True):
    """Route every stand to every runway at every airport, and cache the result."""
    parts = []
    for apt in AIRPORTS:
        a = Airport(apt)
        r = a.routes()
        parts.append(r)
        if verbose:
            print(f"  {apt}: {len(a.stands):4d} stands, {len(a.thresholds):2d} thresholds, "
                  f"{len(a.nodes):5d} nodes -> {len(r):6d} routes", flush=True)
    t = pd.concat(parts, ignore_index=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(TABLE_PATH, index=False)
    return t


def load_table():
    if not TABLE_PATH.exists():
        return build_table(verbose=False)
    return pd.read_parquet(TABLE_PATH)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        raise SystemExit("refetch with the Overpass probe; geometry is cached in dataset/geo/osm/")
    t = build_table()
    print(f"\n{len(t):,} routes written to {TABLE_PATH}")
    print(t.groupby("ADEP_mvt").osm_taxi_dist_m.describe()[["count", "mean", "min", "max"]].round(0).to_string())
