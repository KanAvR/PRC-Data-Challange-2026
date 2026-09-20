"""Context available from provided neighbouring departure reports."""
def add_context(x, extra):
    if len(x) != len(extra) or not x.index.equals(extra.index):
        raise ValueError('context must already be aligned to the query rows')
    added = []
    for direction, total in [('before', 'dep_off_apt_30m'), ('after', 'dep_off_apt_next_30m')]:
        for source in ['AOBT_3_flt', 'LOBT_flt']:
            key = 'proxy_apt_' + source + '_' + direction + '_30'
            x[key] = extra[key]
            x[key + '_coverage'] = (extra[key + '_count'] / x[total].clip(lower=1)).clip(0, 1)
            added += [key, key + '_coverage']
    return added
