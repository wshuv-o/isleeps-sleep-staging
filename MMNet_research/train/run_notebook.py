"""
run_notebook.py -- execute the reproduction notebook cell-by-cell, saving the .ipynb after
EVERY cell so its outputs appear progressively (watchable live in VS Code / Jupyter). This is
genuine in-cell execution: each code cell runs in the kernel and its real output is written
back before the next cell starts.

  KMP_DUPLICATE_LIB_OK=TRUE python revision/run_notebook.py
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if len(sys.argv) < 2:
    sys.exit("usage: run_notebook.py <path-to-notebook.ipynb>  (explicit to avoid clobbering the wrong file)")
NB = sys.argv[1]

nb = nbformat.read(NB, as_version=4)
for c in nb.cells:                       # start clean
    if c.cell_type == "code":
        c.outputs = []; c.execution_count = None
nbformat.write(nb, NB)

client = NotebookClient(nb, timeout=-1, kernel_name="mmnet",
                        resources={"metadata": {"path": ROOT}})
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
done = 0; t0 = time.time()
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        done += 1; t = time.time()
        print(f"[cell {done}/{n_code}] executing...", flush=True)
        try:
            client.execute_cell(cell, i)
        except CellExecutionError as e:
            print(f"  cell {done} raised: {e}", flush=True)
        nbformat.write(nb, NB)           # <-- save after each cell so it is watchable live
        print(f"[cell {done}/{n_code}] done ({time.time()-t:.0f}s, total {(time.time()-t0)/60:.1f}m)", flush=True)
print(f"notebook fully executed in {(time.time()-t0)/60:.1f} min", flush=True)
