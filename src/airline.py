"""Operator code from FLIGHT_mvt, normalised to ICAO.

FLIGHT_mvt mixes IATA and ICAO designators for the same carrier: Lufthansa
appears as both LH400 and DLH400, KLM as KL and KLM. Taking the letters in
front of the flight number split every such carrier into two categories, and
mangled IATA codes that contain a digit (U21234 -> "U", A3626 -> "A").

CALLSIGN_flt would give the ICAO code directly, but it is a _flt column and is
NULL on exactly the tail rows the operator feature exists for. So the mapping is
learned once from rows that have both columns and stored in airline_map.csv;
at feature time only FLIGHT_mvt is needed.

    python src/airline.py    # rebuild airline_map.csv from the training files
"""
import glob
import pathlib

import pandas as pd

MAP_PATH = pathlib.Path(__file__).with_name("airline_map.csv")
# an IATA prefix is mapped only when one ICAO callsign clearly owns it: BA is 65%
# BAW (the rest is BA CityFlyer/Shuttle), while short US registrations such as
# N766DL look like IATA codes and map to nothing in particular
MIN_ROWS = 100
MIN_SHARE = 0.5


def split_prefix(flight):
    """(prefix, is_iata): three leading letters are ICAO, else two chars + digit is IATA."""
    s = flight.astype(str)
    icao = s.str.match(r"[A-Z]{3}")
    iata = ~icao & s.str.match(r"[A-Z0-9]{2}\d")
    prefix = s.str[:3].where(~iata, s.str[:2])
    return prefix, iata.values


def build_map(frames):
    rows = []
    for df in frames:
        prefix, iata = split_prefix(df["FLIGHT_mvt"])
        cs = df["CALLSIGN_flt"].astype(str)
        ok = iata & cs.str.match(r"[A-Z]{3}\d").values
        rows.append(pd.DataFrame({"iata": prefix[ok].values, "icao": cs[ok].str[:3].values}))
    counts = pd.concat(rows).value_counts().rename("n").reset_index()
    counts["total"] = counts.groupby("iata")["n"].transform("sum")
    counts["share"] = (counts["n"] / counts["total"]).round(4)
    best = counts.sort_values("n", ascending=False).drop_duplicates("iata")
    best = best[(best["total"] >= MIN_ROWS) & (best["share"] >= MIN_SHARE)]
    return best.sort_values("total", ascending=False)[["iata", "icao", "total", "share"]]


_MAP = None


def airline_code(flight):
    """(ICAO operator code where it can be resolved, whether FLIGHT_mvt was IATA-form)."""
    global _MAP
    if _MAP is None:
        m = pd.read_csv(MAP_PATH, keep_default_na=False)
        _MAP = dict(zip(m["iata"], m["icao"]))
    prefix, iata = split_prefix(flight)
    code = prefix.where(~iata, prefix.map(_MAP).fillna(prefix))
    return code.where(code.str.len() > 0, "missing"), iata


def flight_key(flight):
    """FLIGHT_mvt with its operator normalised, so LH400 and DLH400 are one flight."""
    code, iata = airline_code(flight)
    s = flight.astype(str)
    rest = s.str[3:].where(~iata, s.str[2:])
    return code + rest


if __name__ == "__main__":
    files = sorted(glob.glob("dataset/training_*.parquet"))
    m = build_map(pd.read_parquet(f, columns=["FLIGHT_mvt", "CALLSIGN_flt"]) for f in files)
    m.to_csv(MAP_PATH, index=False)
    print(f"wrote {len(m)} IATA->ICAO mappings to {MAP_PATH}")
