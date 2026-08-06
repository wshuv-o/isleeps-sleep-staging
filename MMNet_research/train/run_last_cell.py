"""Execute ONLY the last (training-curve) cell of figure_hypnogram.ipynb, live.

To run one cell we still need the kernel to hold the earlier definitions, so we execute the
three setup code cells (imports, model, data+functions) purely to build kernel state, then run
the last cell and capture its output. Every pre-existing cell's stored outputs/execution_count
are snapshotted first and restored afterwards, so nothing above the new cell is reset."""
import copy, nbformat
from nbclient import NotebookClient

P = "revision/figure_hypnogram.ipynb"
nb = nbformat.read(P, as_version=4)
cells = nb.cells
code_idx = [i for i, c in enumerate(cells) if c.cell_type == "code"]
define = code_idx[:3]          # imports, model, data+functions  -> just to build state
last = code_idx[-1]            # the new training-curve cell

# snapshot the current outputs/exec_count of every pre-existing cell (all but the new last one)
orig = {i: (copy.deepcopy(cells[i].get("outputs", [])), cells[i].get("execution_count"))
        for i in code_idx if i != last}

kname = nb.metadata.get("kernelspec", {}).get("name", "python3")
client = NotebookClient(nb, timeout=1800, kernel_name=kname)
with client.setup_kernel():
    for i in define:
        print(f"[state] running setup code cell {i}", flush=True)
        client.execute_cell(cells[i], i)
    print(f"[run] executing last cell {last}", flush=True)
    client.execute_cell(cells[last], last)

# undo the re-run of the setup cells: restore their original stored outputs/exec_count
for i, (o, ec) in orig.items():
    cells[i]["outputs"] = o
    cells[i]["execution_count"] = ec

nbformat.write(nb, P)

errs = [o for o in cells[last].get("outputs", []) if o.get("output_type") == "error"]
print(f"[done] last-cell errors: {len(errs)}")
for o in cells[last].get("outputs", []):
    if o.get("output_type") == "stream":
        print(o["text"][-400:])
    if o.get("output_type") == "error":
        print("\n".join(o["traceback"])[:1200])
