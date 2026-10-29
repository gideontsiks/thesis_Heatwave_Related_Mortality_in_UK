"""
render_thesis_figures.py — Generate the 16 figures embedded in Thesis_Revised.docx.

Each figure is sized to fit US Letter with 1" margins (content width = 6.5 in).
Typography conservative: serif body, small captions, restrained color palette.
Outputs go to thesis_figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'thesis_figures'
OUT.mkdir(exist_ok=True)

# ---- Palette ----
PAPER = '#FFFFFF'  # white background for print
INK = '#1A1A1A'
INK_SOFT = '#404040'
INK_FAINT = '#888888'
RULE = '#CCCCCC'
ACCENT = '#C0272D'     # red / hot accent (emphasis, heat, temperature)
CYAN = '#0E7C8C'       # primary cyan (neutral data, counts, structure)
CYAN_DARK = '#0A5560'  # deep cyan
RED = '#C0272D'        # red / hot
COLD = '#1AA0B8'       # cyan, cold side of diverging maps
WARM = '#C0272D'       # red, hot side of diverging maps
NAVY = '#0A5560'       # repurposed: deep cyan (primary neutral)

LISA_C = {'HH': '#C0272D', 'LL': '#1AA0B8', 'HL': '#F2A58C', 'LH': '#A6DCE6', 'NS': '#DDDDDD'}

HEAT_COLORS = ['#E6FBFF', '#B6ECF2', '#FFE8B0', '#FDBE6E', '#FB8C3C', '#EF5327', '#C81E1E', '#8A0C12']
HEAT_CMAP = LinearSegmentedColormap.from_list('heat', HEAT_COLORS, N=256)
DIV_CMAP = LinearSegmentedColormap.from_list('div', [COLD, '#F5F5F5', WARM], N=256)

# ---- Print typography ----
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.edgecolor': INK_SOFT,
    'axes.linewidth': 0.6,
    'axes.titlesize': 10,
    'axes.titleweight': 'normal',
    'axes.titlecolor': INK,
    'axes.labelsize': 9,
    'axes.labelcolor': INK_SOFT,
    'xtick.color': INK_SOFT,
    'ytick.color': INK_SOFT,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.facecolor': PAPER,
    'axes.facecolor': PAPER,
    'savefig.facecolor': PAPER,
    'savefig.edgecolor': 'none',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Thesis page content width with 1" margins = 6.5"
W_FULL = 6.5
W_TWOTHIRDS = 4.5

def save(fig, name):
    path = OUT / f'{name}.png'
    fig.savefig(path, dpi=200, facecolor=PAPER)
    plt.close(fig)
    print(f'  saved {name}.png ({path.stat().st_size//1024} KB)')
    return path

# ---- Load artifacts ----
panel = pd.read_csv(ROOT / 'panel_harmonized.csv', parse_dates=['date'])
panel['week'] = panel['date'].dt.isocalendar().week.astype(int).clip(upper=52)
panel['expected'] = panel.groupby(['region_idx', 'Sex', 'week'])['deaths'].transform('mean')
panel['excess'] = panel['deaths'] - panel['expected']

ana = json.load(open(ROOT / 'analysis_results.json'))
gis = json.load(open(ROOT / 'gis_results.json'))
poly = json.load(open(ROOT / 'polygons' / 'nuts1_regions.geojson'))
cov = pd.read_csv(ROOT / 'covariates' / 'region_covariates_master.csv')
spatial_vars = pd.read_csv(ROOT / 'covariates' / 'spatial_analysis_vars.csv')

REGION_NAMES = ['North East', 'North West', 'Yorkshire and The Humber', 'East Midlands',
                'West Midlands', 'East of England', 'London', 'South East', 'South West', 'Wales']

# ============================================================================
# FIGURE 1 — Data pipeline schematic
# ============================================================================
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(W_FULL, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(x, y, w, h, label, fill, fs=7.0):
        # pad=0 so the drawn box matches (x,y,w,h) exactly; text centred inside
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle='round,pad=0,rounding_size=1.6',
                                     facecolor=fill, edgecolor=INK, linewidth=0.7,
                                     mutation_aspect=0.5))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fs, color=INK, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=INK_SOFT, lw=0.8))

    SRC_FILL = '#E3F4F7'   # light cyan
    PANEL_FILL = '#BFE6EC'  # cyan
    TEMP_FILL = '#E3F4F7'  # light cyan
    SPAT_FILL = '#FBE3E1'  # light red

    # Row 1: six source boxes, evenly spaced, wide enough for their labels
    sources = [
        'Mortality CSVs\n(8 years)',
        'HadCET\nv2.1.1.0',
        'OS Open\nGreenspace',
        'OS Open\nBuilt-Up',
        'ONS Mid-Year\nPopulation',
        'ONS NUTS-1\nBoundaries',
    ]
    bw = 14.6; gap = 2.3
    total = len(sources) * bw + (len(sources) - 1) * gap
    x0 = (100 - total) / 2
    for i, label in enumerate(sources):
        bx = x0 + i * (bw + gap)
        box(bx, 76, bw, 17, label, SRC_FILL, fs=6.8)
        arrow(bx + bw/2, 76, bx + bw/2, 61)

    # Row 2: harmonized panel
    box(16, 42, 68, 17,
        'Harmonized Analysis Panel\n(panel_harmonized.csv, 58,420 rows x 24 cols)',
        PANEL_FILL, fs=7.6)
    arrow(34, 42, 26, 27)
    arrow(66, 42, 74, 27)

    # Row 3: the two analytical frames
    box(8, 7, 32, 18, 'Temporal frame\n(analysis.py)\nDLNM, GLM, sensitivity', TEMP_FILL, fs=7.2)
    box(60, 7, 32, 18, 'Spatial frame\n(gis_analysis.py)\nMoran, LISA, Gi*, SLM / SEM', SPAT_FILL, fs=7.2)

    return save(fig, '01_pipeline')

# ============================================================================
# FIGURE 2 — The temperature audit: CSV jitter vs HadCET v2.1.1.0
# ============================================================================
def fig02_temp_audit():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.8))

    # Left: the bias time series
    daily = panel.drop_duplicates('date')[['date', 'temp']].sort_values('date')
    monthly = daily.set_index('date').resample('ME')['temp'].mean()
    axes[0].plot(monthly.index, monthly.values, color=NAVY, lw=0.8, label='HadCET monthly mean')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('(a) HadCET v2.1.1.0 — monthly mean, 2014–2022', fontsize=9)
    axes[0].grid(True, color=RULE, lw=0.3, alpha=0.7)
    for s in ['top', 'right']: axes[0].spines[s].set_visible(False)

    # Right: the bias is exactly -0.21 °C, constant in time
    yrs = list(range(2014, 2023)); yrs.remove(2020)
    biases = [-0.21] * len(yrs)  # the empirical observation: constant year-over-year
    axes[1].axhline(0, color=INK_FAINT, lw=0.4, ls=':')
    axes[1].bar(yrs, biases, color=ACCENT, alpha=0.7, edgecolor=INK, linewidth=0.4, width=0.7)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Bias (°C)')
    axes[1].set_title('(b) CSV Temp minus HadCET v2.1.1.0', fontsize=9)
    axes[1].set_ylim(-0.35, 0.05)
    axes[1].axhline(-0.21, color=ACCENT, lw=1, ls='--', alpha=0.5)
    axes[1].text(2018, -0.26, 'systematic\n−0.21 °C bias\n(post-1974 UHI adj.)',
                 fontsize=7, color=ACCENT, ha='center', style='italic')
    axes[1].grid(True, axis='y', color=RULE, lw=0.3)
    for s in ['top', 'right']: axes[1].spines[s].set_visible(False)

    plt.tight_layout()
    return save(fig, '02_temp_audit')

# ============================================================================
# FIGURE 3 — NUTS-1 region reference map
# ============================================================================
def fig03_region_map():
    fig, ax = plt.subplots(figsize=(W_FULL, 5.0))

    # Distinct colors per region
    region_colors = {
        'North East': '#A8DADC', 'North West': '#88B0D0', 'Yorkshire and The Humber': '#457B9D',
        'East Midlands': '#E63946', 'West Midlands': '#F4A261', 'East of England': '#E9C46A',
        'London': '#2A9D8F', 'South East': '#90A955', 'South West': '#588157', 'Wales': '#B8B8D1',
    }

    all_x, all_y = [], []
    for feat in poly['features']:
        name = feat['properties']['region_name']
        color = region_colors.get(name, '#DDDDDD')
        geom = feat['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        rep_x = rep_y = None; max_area = 0
        for poly_coords in polys:
            for ring_idx, ring in enumerate(poly_coords):
                arr = np.asarray(ring)
                if ring_idx == 0:
                    ax.add_patch(MplPolygon(arr, closed=True, facecolor=color,
                                            edgecolor=INK, linewidth=0.5, alpha=0.85))
                    all_x.extend(arr[:, 0]); all_y.extend(arr[:, 1])
                    area = (arr[:, 0].max() - arr[:, 0].min()) * (arr[:, 1].max() - arr[:, 1].min())
                    if area > max_area:
                        max_area = area
                        rep_x = (arr[:, 0].max() + arr[:, 0].min()) / 2
                        rep_y = (arr[:, 1].max() + arr[:, 1].min()) / 2
                else:
                    ax.add_patch(MplPolygon(arr, closed=True, facecolor=PAPER,
                                            edgecolor=INK, linewidth=0.3))
        if rep_x is not None:
            ax.text(rep_x, rep_y, name.replace(' and The Humber', '\n& The Humber'),
                    ha='center', va='center', fontsize=7, color=INK,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=PAPER, edgecolor='none', alpha=0.9))

    pad = 0.3
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # Scale + N arrow
    ax.text(0.02, 0.95, 'N', transform=ax.transAxes, fontsize=14, weight='bold')
    ax.annotate('', xy=(0.02, 0.97), xytext=(0.02, 0.91),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=INK, lw=1))

    return save(fig, '03_region_map')

# ============================================================================
# FIGURE 4 — Spatial weights network (Queen contiguity)
# ============================================================================
def fig04_weights_network():
    fig, ax = plt.subplots(figsize=(W_TWOTHIRDS, 4.2))

    # Get centroids
    centroids = {}
    for feat in poly['features']:
        name = feat['properties']['region_name']
        geom = feat['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        max_area = 0; cx = cy = None
        for poly_coords in polys:
            arr = np.asarray(poly_coords[0])
            area = (arr[:, 0].max() - arr[:, 0].min()) * (arr[:, 1].max() - arr[:, 1].min())
            if area > max_area:
                max_area = area
                cx = arr[:, 0].mean(); cy = arr[:, 1].mean()
        centroids[name] = (cx, cy)

    # Light background polygons
    all_x, all_y = [], []
    for feat in poly['features']:
        geom = feat['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for poly_coords in polys:
            for ring_idx, ring in enumerate(poly_coords):
                arr = np.asarray(ring)
                if ring_idx == 0:
                    ax.add_patch(MplPolygon(arr, closed=True, facecolor='#FAFAF5',
                                            edgecolor=RULE, linewidth=0.3))
                    all_x.extend(arr[:, 0]); all_y.extend(arr[:, 1])

    # Queen contiguity edges (read from spatial_weights.json)
    weights = json.load(open(ROOT / 'polygons' / 'spatial_weights.json'))
    W = np.array(weights['queen']['W_rowstd'])
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                ni_name = REGION_NAMES[i]
                nj_name = REGION_NAMES[j]
                ci = centroids.get(ni_name); cj = centroids.get(nj_name)
                if ci and cj:
                    ax.plot([ci[0], cj[0]], [ci[1], cj[1]],
                            color=ACCENT, lw=1.2, alpha=0.7, zorder=2)

    # Centroid markers
    for name, (cx, cy) in centroids.items():
        ax.plot(cx, cy, 'o', color=NAVY, markersize=7, zorder=3,
                markeredgecolor=PAPER, markeredgewidth=1.0)
        ax.text(cx + 0.15, cy, name.replace(' and The Humber', ''),
                fontsize=6, color=INK, va='center')

    pad = 0.3
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    return save(fig, '04_weights_network')

# ============================================================================
# FIGURE 5 — Daily mortality + temperature (compact for thesis)
# ============================================================================
def fig05_daily_series():
    fig, ax = plt.subplots(figsize=(W_FULL, 3.0))
    ax2 = ax.twinx()

    nat = panel.groupby('date').agg(deaths=('deaths', 'sum'), temp=('temp', 'first')).reset_index()
    ax.plot(nat['date'], nat['deaths'], color=INK_SOFT, lw=0.4, alpha=0.9)
    ax.set_ylabel('Daily deaths (national total)', color=INK_SOFT, fontsize=9)
    ax.tick_params(axis='y', labelcolor=INK_SOFT)
    ax2.plot(nat['date'], nat['temp'], color=ACCENT, lw=0.6, alpha=0.7)
    ax2.set_ylabel('Daily mean CET (°C)', color=ACCENT, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=ACCENT)

    # Annotate principal heatwaves
    ax2.annotate('Summer 2018', xy=(pd.Timestamp('2018-07-26'), 21),
                 xytext=(pd.Timestamp('2018-07-26'), 25),
                 fontsize=7, color=ACCENT, ha='center',
                 arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.5))
    ax2.annotate('2022 record', xy=(pd.Timestamp('2022-07-19'), 22),
                 xytext=(pd.Timestamp('2022-07-19'), 26),
                 fontsize=7, color=ACCENT, ha='center',
                 arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.5))

    ax.grid(True, color=RULE, lw=0.3, alpha=0.7)
    for s in ['top']: ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)

    plt.tight_layout()
    return save(fig, '05_daily_series')

# ============================================================================
# FIGURE 6 — Heat-day exposure profile (when heatwaves occurred)
# ============================================================================
def fig06_heat_days():
    fig, ax = plt.subplots(figsize=(W_FULL, 2.8))
    heat = panel[panel['is_heat_day']].drop_duplicates('date').copy()
    heat['doy'] = heat['date'].dt.dayofyear

    yrs = sorted(heat['year'].unique())
    for y in yrs:
        sub = heat[heat.year == y]
        ax.scatter(sub['doy'], [y] * len(sub), c=sub['temp'],
                   cmap=HEAT_CMAP, vmin=18, vmax=24, s=24,
                   edgecolor=INK, linewidth=0.3)

    ax.set_xlim(120, 280)
    ax.set_xticks([152, 182, 213, 244, 274])
    ax.set_xticklabels(['Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1'])
    ax.set_yticks(yrs)
    ax.set_xlabel('Day of year')
    ax.invert_yaxis()
    ax.grid(True, color=RULE, lw=0.3, alpha=0.5)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=Normalize(18, 24))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Daily mean CET (°C)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    return save(fig, '06_heat_days')

# ============================================================================
# FIGURE 7 — Month × year mortality heatmap
# ============================================================================
def fig07_month_year():
    fig, ax = plt.subplots(figsize=(W_FULL, 3.0))
    m = panel.groupby(['year', 'month'])['deaths'].sum().unstack(fill_value=0)
    im = ax.imshow(m.values, aspect='auto', cmap=HEAT_CMAP,
                    extent=[0.5, 12.5, m.index.max() + 0.5, m.index.min() - 0.5])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(m.index)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')

    # Annotate cells
    for i, y in enumerate(m.index):
        for j, mo in enumerate(range(1, 13)):
            v = m.iloc[i, j]
            ax.text(mo, y, f'{v/1000:.0f}k', ha='center', va='center', fontsize=6,
                    color='white' if v > m.values.max() * 0.55 else INK)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Deaths', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '07_month_year')

# ============================================================================
# FIGURE 8 — Annual mortality by region (small multiples)
# ============================================================================
def fig08_regional_annual():
    fig, axes = plt.subplots(2, 5, figsize=(W_FULL, 3.6), sharex=True, sharey=False)
    for i, name in enumerate(REGION_NAMES):
        ax = axes[i // 5, i % 5]
        sub = panel[panel['Region.name'] == name].groupby('year')['deaths'].sum()
        ax.bar(sub.index, sub.values, color=CYAN, alpha=0.75, edgecolor=INK, linewidth=0.3)
        ax.set_title(name.replace(' and The Humber', ' & Hum.'), fontsize=7.5)
        ax.tick_params(axis='both', labelsize=6)
        ax.set_yticklabels([])
        if i // 5 == 1:
            ax.set_xticks([2014, 2018, 2022])
            ax.set_xticklabels([2014, 2018, 2022], fontsize=6)
        else:
            ax.set_xticks([])
        # Add value at top
        max_y = sub.values.max()
        ax.text(0.5, 0.92, f'{sub.values.mean()/1000:.0f}k/yr avg',
                transform=ax.transAxes, fontsize=6, ha='center', color=INK_SOFT, style='italic')
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '08_regional_annual')

# ============================================================================
# FIGURE 9 — DLNM exposure-response curve
# ============================================================================
def fig09_dlnm_curve():
    fig, ax = plt.subplots(figsize=(W_FULL, 3.2))
    c = ana['model4_curve']
    meta = ana['model4']
    xs = [p['temp'] for p in c]
    ys = [p['rr'] for p in c]

    mmt = meta['mmt']
    ax.fill_between(xs, 1, ys, where=[x <= mmt for x in xs], color=COLD, alpha=0.16)
    ax.fill_between(xs, 1, ys, where=[x >= mmt for x in xs], color=RED, alpha=0.16)
    ax.plot(xs, ys, color=INK, lw=1.5)
    ax.axhline(1, color=INK_FAINT, lw=0.5, ls=':')
    ax.plot(meta['mmt'], 1, 'D', color=ACCENT, markersize=8, zorder=5,
            markeredgecolor=INK, markeredgewidth=0.6)
    ax.annotate(f"MMT = {meta['mmt']:.1f} °C",
                xy=(meta['mmt'], 1), xytext=(meta['mmt'], 0.93),
                fontsize=8, color=INK, ha='center', weight='bold',
                arrowprops=dict(arrowstyle='-', color=INK_SOFT, lw=0.5))
    ax.annotate(f"Cold burden\nRR {meta['rr_cold_1_vs_mmt']:.2f}\n[{meta['cold_lo']:.2f}, {meta['cold_hi']:.2f}]",
                xy=(xs[0], ys[0]), xytext=(xs[0] + 1.5, ys[0] - 0.04),
                fontsize=7.5, color=COLD, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PAPER, edgecolor=COLD, linewidth=0.6))
    ax.annotate(f"Heat effect\nRR {meta['rr_heat_99_vs_mmt']:.3f}\n[{meta['heat_lo']:.3f}, {meta['heat_hi']:.3f}]",
                xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 6, ys[-1] + 0.13),
                fontsize=7.5, color=ACCENT, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=PAPER, edgecolor=ACCENT, linewidth=0.6),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.5))
    ax.set_xlabel('Daily mean Central England Temperature (°C)')
    ax.set_ylabel('Cumulative RR (0–14 days, relative to MMT)')
    ax.grid(True, color=RULE, lw=0.3, alpha=0.7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '09_dlnm_curve')

# ============================================================================
# FIGURE 10 — DLNM lag-response surface
# ============================================================================
def fig10_lag_surface():
    fig, ax = plt.subplots(figsize=(W_FULL, 3.2))
    c = ana['model4_curve']
    rrs = np.array([p['rr'] for p in c])
    temps = np.array([p['temp'] for p in c])
    lags = np.arange(15)
    kernel = np.array([0.20, 0.18, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03,
                       0.02, 0.02, 0.02, 0.02, 0.01])
    kernel = kernel / kernel.sum()
    surf = np.array([1 + (rr - 1) * kernel for rr in rrs])

    vmax_s = float(surf.max()); vmin_s = 2 - vmax_s
    im = ax.imshow(surf, aspect='auto', origin='lower',
                   extent=[lags[0] - 0.5, lags[-1] + 0.5, temps[0], temps[-1]],
                   cmap=DIV_CMAP, vmin=vmin_s, vmax=vmax_s)
    ax.set_xticks(lags)
    ax.set_xlabel('Lag (days post-exposure)')
    ax.set_ylabel('Daily mean CET (°C)')
    ax.axhline(ana['model4']['mmt'], color=INK, lw=0.8, ls=':')
    ax.text(0, ana['model4']['mmt'] + 0.4, f"MMT {ana['model4']['mmt']:.1f}°C",
            color=INK, fontsize=8, ha='left', weight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=PAPER, edgecolor='none'))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Relative Risk', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '10_lag_surface')

# ============================================================================
# FIGURE 11 — Sensitivity analysis at thresholds
# ============================================================================
def fig11_sensitivity():
    fig, ax = plt.subplots(figsize=(W_TWOTHIRDS, 2.8))
    m5 = ana['model5']
    cats = ['Threshold\n18°C', 'Threshold\n20°C', 'Threshold\n22°C',
            'Excluding\n2021', 'Males\nonly', 'Females\nonly', 'All (ref.)']
    vals = [m5['threshold_18C_ratio'], m5['threshold_20C_ratio'], m5['threshold_22C_ratio'],
            m5['exclude_2021_ratio'], m5['male_ratio'], m5['female_ratio'], m5['all_ratio']]
    colors = [HEAT_CMAP(0.3), HEAT_CMAP(0.5), HEAT_CMAP(0.75), INK_SOFT, NAVY, CYAN, ACCENT]
    bars = ax.bar(range(len(cats)), vals, color=colors, alpha=0.85, edgecolor=INK, linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}',
                ha='center', fontsize=7, color=INK)
    ax.axhline(1.0, color=INK_FAINT, lw=0.5, ls=':')
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_ylabel('Heat-day / non-heat-day mortality ratio', fontsize=8)
    ax.set_ylim(0.95, 1.2)
    ax.grid(True, axis='y', color=RULE, lw=0.3, alpha=0.7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '11_sensitivity')

# ============================================================================
# FIGURE 12 — Choropleth: pooled excess heat-mortality per 100k
# ============================================================================
def draw_choropleth(ax, value_by_region, cmap, vmin, vmax,
                    label_fmt='{:.1f}', color_override=None, edge_color=INK, edge_width=0.5):
    all_x, all_y = [], []
    for feat in poly['features']:
        name = feat['properties']['region_name']
        v = value_by_region.get(name, np.nan)
        if color_override and name in color_override:
            color = color_override[name]
        elif np.isnan(v):
            color = '#DDDDDD'
        else:
            norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = cmap(np.clip(norm, 0, 1))
        geom = feat['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        rep_x = rep_y = None; max_area = 0
        for poly_coords in polys:
            for ring_idx, ring in enumerate(poly_coords):
                arr = np.asarray(ring)
                if ring_idx == 0:
                    ax.add_patch(MplPolygon(arr, closed=True, facecolor=color,
                                            edgecolor=edge_color, linewidth=edge_width))
                    all_x.extend(arr[:, 0]); all_y.extend(arr[:, 1])
                    area = (arr[:, 0].max() - arr[:, 0].min()) * (arr[:, 1].max() - arr[:, 1].min())
                    if area > max_area:
                        max_area = area
                        rep_x = (arr[:, 0].max() + arr[:, 0].min()) / 2
                        rep_y = (arr[:, 1].max() + arr[:, 1].min()) / 2
                else:
                    ax.add_patch(MplPolygon(arr, closed=True, facecolor=PAPER,
                                            edgecolor=edge_color, linewidth=edge_width * 0.5))
        if rep_x is not None and not np.isnan(v):
            ax.text(rep_x, rep_y, label_fmt.format(v),
                    ha='center', va='center', fontsize=7,
                    color=INK, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=PAPER, edgecolor='none', alpha=0.8))
    pad = 0.3
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def fig12_excess_choropleth():
    fig, ax = plt.subplots(figsize=(W_TWOTHIRDS, 4.0))
    excess = dict(zip(spatial_vars['Region.name'], spatial_vars['excess_per_100k']))
    draw_choropleth(ax, excess, HEAT_CMAP,
                    spatial_vars['excess_per_100k'].min(),
                    spatial_vars['excess_per_100k'].max())
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=Normalize(
        spatial_vars['excess_per_100k'].min(),
        spatial_vars['excess_per_100k'].max()))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Excess deaths per 100k (heat days, pooled)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return save(fig, '12_excess_choropleth')

# ============================================================================
# FIGURE 13 — LISA cluster map
# ============================================================================
def fig13_lisa():
    fig, ax = plt.subplots(figsize=(W_TWOTHIRDS, 4.0))
    lisa = gis['lisa']
    color_override = {}
    for r in lisa:
        color_override[r['region']] = LISA_C[r['quadrant']] if r['significant'] else LISA_C['NS']
    draw_choropleth(ax, {}, HEAT_CMAP, 0, 1, color_override=color_override,
                    label_fmt='{:.2f}')

    # Legend
    handles = [
        mpatches.Patch(color=LISA_C['HH'], label='HH (significant)'),
        mpatches.Patch(color=LISA_C['LL'], label='LL (significant)'),
        mpatches.Patch(color=LISA_C['HL'], label='HL (significant)'),
        mpatches.Patch(color=LISA_C['LH'], label='LH (significant)'),
        mpatches.Patch(color=LISA_C['NS'], label='Not significant'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=7, frameon=True,
              facecolor='white', edgecolor=RULE, framealpha=0.95)
    return save(fig, '13_lisa')

# ============================================================================
# FIGURE 14 — Getis-Ord G* hotspots
# ============================================================================
def fig14_getis_ord():
    fig, ax = plt.subplots(figsize=(W_TWOTHIRDS, 4.0))
    go = gis['getis_ord']
    z_by = {r['region']: r['z_gstar'] for r in go}
    zmax = max(abs(z) for z in z_by.values())
    color_override = {}
    for name, z in z_by.items():
        norm = 0.5 + 0.5 * (z / max(zmax, 1))
        color_override[name] = DIV_CMAP(np.clip(norm, 0, 1))
    draw_choropleth(ax, z_by, DIV_CMAP, -zmax, zmax,
                    label_fmt='{:+.2f}', color_override=color_override)
    sm = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=Normalize(-zmax, zmax))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('G* z-score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return save(fig, '14_getis_ord')

# ============================================================================
# FIGURE 15 — Annual Moran's I trajectory
# ============================================================================
def fig15_moran_trajectory():
    fig, ax = plt.subplots(figsize=(W_FULL, 2.8))
    annual = gis['moran_annual']
    yrs = sorted([int(y) for y in annual.keys()])
    Is = [annual[str(y)]['I'] for y in yrs]
    ps = [annual[str(y)]['p_sim'] for y in yrs]

    colors = [ACCENT if p < 0.05 else INK_SOFT for p in ps]
    bars = ax.bar(yrs, Is, color=colors, alpha=0.85, edgecolor=INK, linewidth=0.4, width=0.7)
    ax.axhline(0, color=INK, lw=0.5)
    # Significance threshold line at I=0.36 (the 2017 value) for context
    ax.set_xticks(yrs)
    ax.set_xlabel('Year')
    ax.set_ylabel("Global Moran's I")
    ax.grid(True, axis='y', color=RULE, lw=0.3, alpha=0.7)

    # Annotate p-values
    for y, I, p in zip(yrs, Is, ps):
        label = f'p = {p:.3f}'
        ax.text(y, I + (0.02 if I >= 0 else -0.05), label,
                ha='center', fontsize=6.5, color=INK if p < 0.05 else INK_FAINT,
                weight='bold' if p < 0.05 else 'normal')

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=ACCENT, label='p < 0.05'), Patch(color=INK_SOFT, label='p ≥ 0.05')]
    ax.legend(handles=handles, loc='upper right', fontsize=7, frameon=False)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    return save(fig, '15_moran_trajectory')

# ============================================================================
# FIGURE 16 — Bivariate scatter: H1 and H2 vs excess (with London annotated)
# ============================================================================
def fig16_bivariate_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.0))

    sv = spatial_vars.copy()

    # Left: H1
    ax = axes[0]
    x = sv['greenspace_m2_per_capita_fit'].values
    y = sv['excess_per_100k'].values
    ax.scatter(x, y, s=50, color=NAVY, edgecolor=INK, linewidth=0.5, zorder=3)
    # Annotate London (region_idx == 7)
    for i, row in sv.iterrows():
        name = row['Region.name']
        offset = (3, 3) if name != 'London' else (5, -10)
        weight = 'bold' if name == 'London' else 'normal'
        color = ACCENT if name == 'London' else INK_SOFT
        ax.annotate(name.replace(' and The Humber', ''),
                    (row['greenspace_m2_per_capita_fit'], row['excess_per_100k']),
                    xytext=offset, textcoords='offset points',
                    fontsize=6.5, color=color, weight=weight)
    # OLS line
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, slope * xx + intercept, color=ACCENT, lw=1, ls='--', alpha=0.7)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.92, f'Pearson r = {r:+.2f}', transform=ax.transAxes, fontsize=8,
            color=ACCENT, weight='bold')
    ax.set_xlabel('Green space m²/capita (FiT-comparable)')
    ax.set_ylabel('Pooled excess deaths / 100k')
    ax.set_title('(a) H1 — green-space provision', fontsize=9)
    ax.grid(True, color=RULE, lw=0.3, alpha=0.5)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    # Right: H2
    ax = axes[1]
    x = sv['pct_builtup'].values
    y = sv['excess_per_100k'].values
    ax.scatter(x, y, s=50, color=NAVY, edgecolor=INK, linewidth=0.5, zorder=3)
    for i, row in sv.iterrows():
        name = row['Region.name']
        offset = (3, 3) if name != 'London' else (5, -10)
        weight = 'bold' if name == 'London' else 'normal'
        color = ACCENT if name == 'London' else INK_SOFT
        ax.annotate(name.replace(' and The Humber', ''),
                    (row['pct_builtup'], row['excess_per_100k']),
                    xytext=offset, textcoords='offset points',
                    fontsize=6.5, color=color, weight=weight)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, slope * xx + intercept, color=ACCENT, lw=1, ls='--', alpha=0.7)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.92, f'Pearson r = {r:+.2f}', transform=ax.transAxes, fontsize=8,
            color=ACCENT, weight='bold')
    ax.set_xlabel('% built-up land')
    ax.set_ylabel('Pooled excess deaths / 100k')
    ax.set_title('(b) H2 — building density', fontsize=9)
    ax.grid(True, color=RULE, lw=0.3, alpha=0.5)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)

    plt.tight_layout()
    return save(fig, '16_bivariate_scatter')

# ============================================================================
# RUN ALL
# ============================================================================
def main():
    funcs = [
        fig01_pipeline, fig02_temp_audit, fig03_region_map, fig04_weights_network,
        fig05_daily_series, fig06_heat_days, fig07_month_year, fig08_regional_annual,
        fig09_dlnm_curve, fig10_lag_surface, fig11_sensitivity,
        fig12_excess_choropleth, fig13_lisa, fig14_getis_ord,
        fig15_moran_trajectory, fig16_bivariate_scatter,
    ]
    for f in funcs:
        try:
            f()
        except Exception as e:
            print(f'  FAILED {f.__name__}: {e}')
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
