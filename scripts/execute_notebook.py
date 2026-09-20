"""
Executes the audited Pinnacle Plus Capstone Jupyter Notebook from start to finish,
validating every cell and saving the executed outputs.
"""

import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, 'notebooks', 'pinnacle_plus_capstone.ipynb')

def run_notebook():
    print("\n" + "=" * 80)
    print("      EXECUTING PINNACLE PLUS AUDITED JUPYTER NOTEBOOK")
    print("=" * 80)
    print(f"Target Notebook: {NOTEBOOK_PATH}")

    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    print(f"Total cells: {total_cells} (Code cells: {len(code_cells)})")

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    t0 = time.time()
    try:
        notebook_dir = os.path.join(BASE_DIR, 'notebooks')
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        elapsed = time.time() - t0
        print(f"\nSUCCESS: All code cells executed cleanly in {elapsed:.1f}s.")

        # Save executed notebook
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Executed outputs saved back to: {NOTEBOOK_PATH}")
        return True

    except Exception as e:
        print(f"\nERROR during notebook execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_notebook()
    sys.exit(0 if success else 1)
