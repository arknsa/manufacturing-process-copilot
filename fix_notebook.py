import sys, json
sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'd:/Kuliah/Project/manufacturing-factory-simulation/manufacturing-process-copilot/ml/notebooks/05_feature_engineering.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove duplicate f003b cells — keep only the last one (most recently inserted)
seen_ids = {}
for i, cell in enumerate(nb['cells']):
    cid = cell['id']
    if cid in seen_ids:
        seen_ids[cid].append(i)
    else:
        seen_ids[cid] = [i]

# Find duplicate f003b and remove the EARLIER one (lower index)
duplicate_indices_to_remove = []
for cid, positions in seen_ids.items():
    if len(positions) > 1:
        print(f"Duplicate cell '{cid}' at positions {positions} — removing {positions[:-1]}")
        duplicate_indices_to_remove.extend(positions[:-1])

# Remove duplicates (reverse order to preserve indices)
for idx in sorted(duplicate_indices_to_remove, reverse=True):
    del nb['cells'][idx]

print(f"After dedup: {len(nb['cells'])} cells")
print("IDs:", [c['id'] for c in nb['cells']])

# Find f003b cell and fix the datetime parsing
idx_map = {c['id']: i for i, c in enumerate(nb['cells'])}
f003b = nb['cells'][idx_map['f003b']]

# Fix: replace pd.to_datetime(tr['planned_start']) with format='ISO8601'
new_source = []
for line in f003b['source']:
    if "pd.to_datetime(tr['planned_start'])" in line:
        line = line.replace(
            "pd.to_datetime(tr['planned_start'])",
            "pd.to_datetime(tr['planned_start'], format='ISO8601')"
        )
        print(f"Fixed datetime line: {line.rstrip()}")
    new_source.append(line)
f003b['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook fixed and saved.")
