hist_time = slice('1950', '2014')
future_time = slice('2015', '2120')
chunks = {'time': -1, 'x': 50, 'y': 50}
# xy_region = {'x': slice(0, 100), 'y': slice(0, 100)}
xy_region = None

models = [
    ['CanESM5', 'historical', 'r10i1p1f1'],
    ['CanESM5', 'ssp245', 'r10i1p1f1'],
    ['CanESM5', 'ssp370', 'r10i1p1f1'],
    ['CanESM5', 'ssp585', 'r10i1p1f1'],
    ['FGOALS-g3', 'historical', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp245', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp370', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp585', 'r1i1p1f1'],
    ['HadGEM3-GC31-LL', 'historical', 'r1i1p1f3'],
    ['HadGEM3-GC31-LL', 'ssp245', 'r1i1p1f3'],
    ['MIROC-ES2L', 'historical', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp245', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp370', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp585', 'r1i1p1f2'],
    ['MIROC6', 'historical', 'r10i1p1f1'],
    ['MIROC6', 'ssp245', 'r10i1p1f1'],
    ['MIROC6', 'ssp585', 'r10i1p1f1'],
    ['MRI-ESM2-0', 'historical', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp245', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp370', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp585', 'r1i1p1f1'],
    ['UKESM1-0-LL', 'historical', 'r10i1p1f2'],
    ['UKESM1-0-LL', 'ssp245', 'r10i1p1f2'],
    ['UKESM1-0-LL', 'ssp370', 'r10i1p1f2'],
]
