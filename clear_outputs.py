import json
from pathlib import Path

nb_path = Path('SGD Classifier_prevyears.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# First, let's see what cells we have around lines 1857+
target_cells = []
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        # Print first few code cells to find our targets
        if 'execution_count' in cell:
            exec_count = cell['execution_count']
            if exec_count in [5, None]:  # Section 16 and later
                print(f"Code cell {i}: id={cell.get('id')}, exec_count={exec_count}, has outputs={len(cell.get('outputs', []))>0}")

cell_ids = ['VSC-b54e7896', 'VSC-5ccaa359', 'VSC-19fd26bd', 'VSC-c2dc332e']

print("BEFORE clearing outputs:")
for cell_id in cell_ids:
    cell = next((c for c in nb['cells'] if c['id'] == cell_id), None)
    if cell:
        print(f'  Cell {cell_id}: execution_count={cell.get("execution_count")}, outputs count={len(cell.get("outputs", []))}')
    else:
        print(f'  Cell {cell_id}: NOT FOUND')

# Clear outputs and execution counts
for cell_id in cell_ids:
    cell = next((c for c in nb['cells'] if c['id'] == cell_id), None)
    if cell:
        cell['outputs'] = []
        cell['execution_count'] = None

print("\nAFTER clearing outputs:")
for cell_id in cell_ids:
    cell = next((c for c in nb['cells'] if c['id'] == cell_id), None)
    if cell:
        print(f'  Cell {cell_id}: execution_count={cell.get("execution_count")}, outputs count={len(cell.get("outputs", []))}')

# Write back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nNotebook saved with cleared outputs.")
