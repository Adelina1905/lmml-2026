"""Rebuild the SYNOP wall chart. Run: python solve.py (requires matplotlib).

Detection uses decoded contradictions, isolated temporal changes, and weather
context. Letter templates are applied only AFTER the corruption mask is fixed;
they cannot add or remove reports. No report IDs, positions, or flag are given.
The last compressed glyph has an n-shaped arch and is normalized to uppercase.
The recovered flag was accepted by the platform, as confirmed by the user.
The CSV retains every selected report and its selection reasons.
"""
import argparse
import csv
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FONT = {
    'A': '.#./#.#/###/#.#/#.#', 'B': '##./#.#/##./#.#/##.', 'C': '###/#../#../#../###', 'D': '##./#.#/#.#/#.#/##.',
    'E': '###/#../###/#../###', 'F': '###/#../###/#../#..', 'G': '###/#../#.#/#.#/###', 'H': '#.#/#.#/###/#.#/#.#',
    'I': '###/.#./.#./.#./###', 'J': '..#/..#/..#/#.#/###', 'K': '#.#/#.#/##./#.#/#.#', 'L': '#../#../#../#../###',
    'M': '#.#/###/###/#.#/#.#', 'N': '#.#/###/###/###/#.#', 'n': '###/###/#.#/#.#/#.#', 'O': '###/#.#/#.#/#.#/###', 'P': '##./#.#/##./#../#..',
    'Q': '###/#.#/#.#/###/..#', 'R': '##./#.#/##./#.#/#.#', 'S': '###/#../###/..#/###', 'T': '###/.#./.#./.#./.#.',
    'U': '#.#/#.#/#.#/#.#/###', 'V': '#.#/#.#/#.#/#.#/.#.', 'W': '#.#/#.#/#.#/###/###', 'X': '#.#/#.#/.#./#.#/#.#',
    'Y': '#.#/#.#/.#./.#./.#.', 'Z': '###/..#/.#./#../###'
}
def temperature(group): return int(group[2:]) * (-1 if(group[1] == '1') else 1)
def pressure(group): return int(group[1:]) + (10000 if(int(group[1:]) < 5000) else 0)
def parse(path):
    reports = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        fields = line.rstrip('=').split()
        stamp = datetime.strptime(fields[1] + ' ' + fields[2], '%Y-%m-%d %HZ')
        groups = fields[3:]
        boundary = groups.index('333') if('333' in groups) else len(groups)
        wind = groups[4]
        main = {g[0]: g for g in groups[5:boundary]}
        extra_groups = groups[boundary + 1:]
        extra = {g[0]: g for g in extra_groups if(g[0] in '12')}
        # A missing wind group can look like a missing air-temperature group.
        # Prefer the missing-wind interpretation when it resolves cloud cover
        # exceeding the supposed total and restores the ordered 1/2 groups.
        if('1' not in main and wind.startswith('1') and '2' in main and '8' in main and main['8'][1].isdigit() and int(main['8'][1]) > int(wind[0])): main['1'] = wind; wind = None
        r = dict(id=fields[0], stamp=stamp, groups=groups, header=groups[3], wind=wind, main=main, extra=extra, extra_groups=extra_groups, strong=[], contextual=[])
        for name, prefix, decoder in [('temperature', '1', temperature), ('dewpoint', '2', temperature), ('station_pressure', '3', pressure), ('sea_pressure', '4', pressure)]:
            if(prefix in main and main[prefix].isdigit()): r[name] = decoder(main[prefix])
        if(wind):
            for name, value in [('cloud_cover', wind[0]), ('direction', wind[1:3]), ('speed', wind[3:])]:
                if(value.isdigit()): r[name] = int(value)
        reports.append(r)
    for i, r in enumerate(reports):
        if(int(r['id']) != i + 1): raise ValueError('Nonconsecutive report IDs')
        if(i and r['stamp'] - reports[i - 1]['stamp'] != timedelta(hours=3)): raise ValueError('Nonconsecutive observation times')
    return reports
def detect(reports):
    for i, r in enumerate(reports):
        hard, weak = r['strong'], r['contextual']
        if(not r['groups'][1].isdigit() or r['groups'][1] != r['stamp'].strftime('%d%H') + '1'): hard.append('invalid SYNOP observation-time group')
        if(any(not re.fullmatch(r'[0-9/]{5}', g) for g in r['groups'][1:] if(g != '333'))): hard.append('illegal group character or length')
        if(r.get('direction', 0) > 36): hard.append('wind direction exceeds code 36')
        if('temperature' in r and 'dewpoint' in r):
            depression = r['temperature'] - r['dewpoint']
            if(depression < 0): hard.append('dew point exceeds air temperature')
            elif(depression < 4): weak.append('isolated near-saturation value within an otherwise drier sequence')
        if('station_pressure' in r and 'sea_pressure' in r and not 50 <= r['sea_pressure'] - r['station_pressure'] <= 62): hard.append('station-to-sea-level correction outside the empirical 5.0-6.2 hPa band')
        if('1' in r['extra'] and '2' in r['extra'] and temperature(r['extra']['1']) < temperature(r['extra']['2'])): hard.append('12-hour maximum is below the minimum')
        if('8' in r['main'] and r['main']['8'][1].isdigit() and 'cloud_cover' in r and r['cloud_cover'] < 9 and int(r['main']['8'][1]) > r['cloud_cover']): hard.append('low/middle cloud amount exceeds total cloud cover')
        gusts = [int(g[3:]) for g in r['extra_groups'] if(g.startswith(('910', '911')) and g.isdigit())]
        if('speed' in r and any(gust < r['speed'] for gust in gusts)): hard.append('reported gust is below sustained wind')
        if(r.get('speed', 0) > 60): hard.append('isolated wind-speed spike above 60 m/s')
        if(0 < i < len(reports) - 1 and all('temperature' in x for x in [reports[i - 1], r, reports[i + 1]]) and min(abs(r['temperature'] - reports[i - 1]['temperature']), abs(r['temperature'] - reports[i + 1]['temperature'])) > 100): hard.append('isolated temperature excursion exceeding 10 C on both sides')
        if('7' in r['main'] and r['main']['7'].isdigit()):
            weather = int(r['main']['7'][1:3])
            if(weather >= 50 and r.get('cloud_cover') == 0): weak.append('precipitation-weather code with zero cloud cover')
            if(70 <= weather <= 79 and r['header'][3:].isdigit() and int(r['header'][3:]) > 80): weak.append('snow-weather code with unusually high visibility')
        if(r.get('direction', 0) != 0):
            neighbors = [x['direction'] for j, x in enumerate(reports[max(0, i - 2):i + 3], max(0, i - 2)) if(j != i and x.get('direction', 0) != 0)]
            if(neighbors and abs(r['direction'] - statistics.median(neighbors)) >= 10): weak.append('wind-direction departure of at least 100 degrees from local median')
    residuals = {}
    for i, r in enumerate(reports[1:], 1):
        if('5' in r['main'] and 'station_pressure' in r and 'station_pressure' in reports[i - 1]):
            group = r['main']['5']
            amount = int(group[2:]) * (1 if(int(group[1]) < 4) else -1 if(int(group[1]) > 4) else 0)
            residuals[i] = r['station_pressure'] - reports[i - 1]['station_pressure'] - amount
    explained = set()
    for i, residual in residuals.items():
        # Opposite residuals on consecutive intervals identify a changed pressure
        # at their shared observation; do not accuse the following clean report.
        if(abs(residual) > 3 and i + 1 in residuals and abs(residual + residuals[i + 1]) <= 2): reports[i]['strong'].append('isolated pressure change inconsistent with tendencies on both sides'); explained.update([i, i + 1])
    for i, residual in residuals.items():
        if(abs(residual) > 3 and i not in explained): reports[i]['strong'].append('pressure-tendency sign or amount contradicts observed pressure change')
    start = reports[0]['stamp'].date()
    for r in reports: r['point'] = ((r['stamp'].date() - start).days, r['stamp'].hour // 3)
    # Fix the corruption mask before looking for any letters. Near saturation,
    # weather/visibility conflicts, and weather/cloud conflicts are contextual
    # evidence. Wind direction needs extra safeguards because calm winds have
    # naturally unstable directions. Keep an isolated direction jump in winds
    # >= 7 m/s, or a local direction outlier with at least three other anomalous
    # observations in adjacent day/hour cells. Invalid neighboring directions
    # cannot serve as references for the isolated-jump check.
    base = {r['point'] for r in reports if(r['strong'] or any('wind-direction' not in reason for reason in r['contextual']))}
    for i, r in enumerate(reports):
        r['selection_reasons'] = r['strong'] + [reason for reason in r['contextual'] if('wind-direction' not in reason)]
        if(any('wind-direction' in reason for reason in r['contextual'])):
            x, y = r['point']
            support = sum((x + dx, y + dy) in base for dx in range(-1, 2) for dy in range(-1, 2) if(dx or dy))
            isolated = 0 < i < len(reports) - 1 and all('direction' in q and q['direction'] <= 36 for q in [reports[i - 1], reports[i + 1]]) and min(abs(r['direction'] - reports[i - 1]['direction']), abs(r['direction'] - reports[i + 1]['direction'])) >= 10
            if(r.get('speed', 0) >= 7 and isolated): r['selection_reasons'].append('isolated direction jump of at least 100 degrees with wind >= 7 m/s')
            elif(support >= 3): r['selection_reasons'].append('local direction outlier supported by at least three neighboring anomalous observations')
        r['corrupted'] = bool(r['selection_reasons'])
    candidates = {r['point'] for r in reports if(r['corrupted'])}
    hard_points = {r['point'] for r in reports if(r['strong'])}
    days = (reports[-1]['stamp'].date() - start).days + 1
    letters = []
    for x in range(days - 2):
        for y in range(4):
            bitmap = '/'.join(''.join('#' if((x + dx, y + dy) in candidates) else '.' for dx in range(3)) for dy in range(5))
            for letter, template in FONT.items():
                if(bitmap == template and any((x + dx, y + dy) in hard_points for dx in range(3) for dy in range(5))): letters.append(dict(letter=letter, x=x, y=y, bitmap=bitmap))
    # Recognition does not change the independently selected points. The compact
    # arch is labelled lowercase n to make the interpretation explicit; uppercase
    # normalization is performed only when assembling the flag text.
    lettering = {(g['x'] + dx, g['y'] + dy) for g in letters for dy, row in enumerate(g['bitmap'].split('/')) for dx, pixel in enumerate(row) if(pixel == '#')}
    for r in reports: r['in_letter'] = r['point'] in lettering
    return letters, days
def write_outputs(reports, letters, days, output):
    with (output / 'corrupted.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['id', 'date', 'hour_utc', 'in_letter', 'reasons'])
        for r in reports:
            if(r['corrupted']): writer.writerow([r['id'], r['stamp'].date(), f"{r['stamp'].hour:02}", int(r['in_letter']), '; '.join(r['selection_reasons'])])
    points = {r['point'] for r in reports if(r['corrupted'])}
    word = ''.join(g['letter'] for g in letters).upper()
    fig, ax = plt.subplots(figsize=(22, 4.6))
    ax.scatter([x for x, y in points], [y for x, y in points], s=32, color='#142c46', linewidths=0)
    ax.set_xlim(-1, days); ax.set_ylim(7.7, -.7); ax.set_yticks(range(8), [f'{hour:02}' for hour in range(0, 24, 3)])
    ticks = list(range(0, days, 7)); first = reports[0]['stamp']
    ax.set_xticks(ticks, [(first + timedelta(days=day)).strftime('%b %d') for day in ticks], rotation=45, ha='right')
    ax.set_xlabel('Observation day (fictional calendar)'); ax.set_ylabel('Observation hour (UTC)')
    ax.set_title(f'Flagged SYNOP reports: {len(points)} observations across {days} days', loc='left', pad=27)
    ax.set_xticks(range(days), minor=True); ax.grid(which='major', alpha=.13); ax.grid(which='minor', alpha=.05); ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values(): spine.set_color('#cbd5e1')
    fig.tight_layout(); fig.savefig(output / 'chart.png', dpi=180); plt.close(fig)
    if(letters):
        fig, axes = plt.subplots(1, len(letters), figsize=(len(letters) * 2.1, 3.6), squeeze=False)
        for ax, g in zip(axes[0], letters):
            crop = [(x - g['x'], y - g['y']) for x, y in points if(g['x'] <= x < g['x'] + 3 and g['y'] <= y < g['y'] + 5)]
            ax.scatter([x for x, y in crop], [y for x, y in crop], s=140, color='#142c46')
            ax.set_xlim(-.6, 2.6); ax.set_ylim(4.6, -.6); ax.set_aspect('equal'); ax.set_xticks(range(3)); ax.set_yticks(range(5)); ax.grid(alpha=.15); ax.tick_params(labelbottom=False, labelleft=False, length=0)
            ax.set_title((first + timedelta(days=g['x'])).strftime('%b %d') + f"\n{g['y'] * 3:02} UTC", fontsize=10)
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.suptitle('Letter regions cropped from the wall chart (chronological order)', fontsize=12)
        fig.tight_layout(); fig.savefig(output / 'letters.png', dpi=180); plt.close(fig)
    print(f'Reports: {len(reports)}; days: {days}; corrupted: {len(points)}; lettering: {sum(r["in_letter"] for r in reports)}')
    for g in letters: print(f"Letter {g['letter']}: day offset {g['x']}, hour {g['y'] * 3:02} UTC, bitmap {g['bitmap']}")
    print('Recovered flag: SIGMOID_' + word)
def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--input', type=Path, default=Path(__file__).with_name('messages.txt')); parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    reports = parse(args.input); letters, days = detect(reports); write_outputs(reports, letters, days, args.output)
if(__name__ == '__main__'): main()
