"""Compare candidate components against a stronger, v15-like local control."""
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from model import combine, V_HI_THR, V_HI_RATIO
from constraints import project_lobt
from normal_correction import apply_normal_correction
from validate import compare

out = Path('output/experiments/neighbor_proxy')
catpath = Path('output/experiments/aobt_residual/cat_ctr1_weather_inner.parquet')
if not catpath.exists():
    raise SystemExit('CatBoost control predictions are not ready yet; no comparison made.')
base = pd.read_parquet('output/experiments/route_delay/weather_base_inner.parquet')
cat = pd.read_parquet(catpath)
neighbor = pd.read_parquet(out / 'weather_inner.parquet')
anchor = pd.read_parquet(out / 'weather_anchor_inner.parquet')
expert = pd.read_parquet('output/experiments/airport_experts/LIRF_weather_inner.parquet')
old_tail = pd.read_parquet('output/experiments/tail_aux_labels/control_inner.parquet')
new_tail = pd.read_parquet('output/experiments/tail_neighbor/inner.parquet')
assert all(base.id.equals(f.id) for f in [cat, neighbor, anchor, expert, old_tail, new_tail])
# Use the same paired classifier recipe in every arm, so numerical differences
# between the earlier frozen classifier fits cannot enter the comparison.
base.loc[~base.has_lobt, 'p'] = old_tail.loc[~base.has_lobt, 'p']
strong = base.copy()
strong['mu'] = .5 * cat.mu + .5 * base.mu
strong['pred'] = project_lobt(combine(strong.mu.to_numpy(), base.p.to_numpy(), base.v.to_numpy()), base.gap_LOBT_flt.to_numpy())
high = base.gap_LOBT_flt > 6000
strong.loc[high, 'pred'] = base.loc[high, 'pred']
rome = base.ADEP_mvt.eq('LIRF')
tail = rome & ~base.has_lobt & (base.v > 6000)
tail_value = base.v.where(base.v < V_HI_THR,base.v*V_HI_RATIO)
def candidate(new_mu):
    new_mu = np.asarray(new_mu)
    r = strong.copy()
    r['mu'] = strong.mu.to_numpy() + .25 * (new_mu - base.mu.to_numpy())
    r['pred'] = apply_normal_correction(strong.pred, base.gap_LOBT_flt,
        base.gap_SCHED_TIME_UTC_mvt, base.p, base.mu, new_mu)
    # Apply the same paired Rome tail correction to every arm.
    r.loc[tail, 'pred'] += .5 * (new_tail.loc[tail, 'p'] - old_tail.loc[tail, 'p']) * (tail_value.loc[tail] - r.loc[tail, 'mu'])
    r['pred'] = r.pred.clip(lower=0)
    return r
control = candidate(neighbor.mu)
p15 = base.p.copy()
p15.loc[rome & ~base.has_lobt] = .5 * old_tail.loc[rome & ~base.has_lobt, 'p'] + .5 * new_tail.loc[rome & ~base.has_lobt, 'p']
local_mu = 2 * expert.mu - neighbor.mu
with_rome = neighbor.mu.copy()
with_rome.loc[rome] = .5 * neighbor.loc[rome, 'mu'] + .5 * local_mu.loc[rome]
control.to_parquet(out / 'strong_corrected_v15_control_inner.parquet', index=False)
for name, mu in [('anchor', anchor.mu), ('rome', with_rome)]:
    # Transfer from the v15-like prediction, preserving boundaries it already
    # reaches, exactly as the next production correction would do.
    r = control.copy()
    r['mu'] = control.mu + .25 * (mu - neighbor.mu)
    r['pred'] = apply_normal_correction(control.pred, base.gap_LOBT_flt,
        base.gap_SCHED_TIME_UTC_mvt, p15, neighbor.mu, mu)
    r.to_parquet(out / ('strong_corrected_' + name + '_inner.parquet'), index=False)
    compare('inner', r, control, 'strong-v15-plus-' + name)
    if name == 'anchor':
        anchor_result = r.copy()

# A conventional 20% new-member blend, limited to matched normal rows where
# the current prediction is directly the normal mean (before projection).
# Missing-LOBT and specialist predictions remain exactly unchanged.
matched_rome = rome & base.has_lobt & (base.gap_LOBT_flt <= 6000)
for name, start in [('rome_member', control), ('anchor_rome_member', anchor_result)]:
    member_mu = start.pred.copy()
    member_mu.loc[matched_rome] = local_mu.loc[matched_rome]
    member = start.copy()
    member['pred'] = apply_normal_correction(start.pred, base.gap_LOBT_flt,
        base.gap_SCHED_TIME_UTC_mvt, p15, start.pred, member_mu, weight=.2)
    member.to_parquet(out / ('strong_corrected_' + name + '_inner.parquet'), index=False)
    compare('inner', member, control, 'strong-v15-plus-' + name)
