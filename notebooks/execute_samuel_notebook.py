"""Execute Samuel's notebook and persist outputs while long cells run."""
import json
import os
from pathlib import Path
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/Samuel_DQN_experiments.ipynb'
STATUS = ROOT / 'notebooks/Samuel_notebook_execution.json'
os.chdir(ROOT)


def save(notebook):
    temporary = NOTEBOOK.with_suffix('.running.tmp')
    nbformat.write(notebook, temporary)
    temporary.replace(NOTEBOOK)


class SavingClient(NotebookClient):
    last_saved = 0

    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if time.monotonic() - self.last_saved > 10:
            save(self.nb)
            self.last_saved = time.monotonic()
        return result


notebook = nbformat.read(NOTEBOOK, as_version=4)

def cell_started(cell_index, **kwargs):
    STATUS.write_text(json.dumps(dict(phase='running', cell_index=cell_index, pid=os.getpid()), indent=2))
    print(f'Executing cell {cell_index}', flush=True)


def cell_finished(**kwargs):
    save(notebook)


try:
    client = SavingClient(notebook, timeout=None, kernel_name='python3',
                          resources={'metadata': {'path': str(ROOT)}},
                          on_cell_start=cell_started, on_cell_executed=cell_finished)
    client.execute()
    STATUS.write_text(json.dumps(dict(phase='completed', pid=os.getpid()), indent=2))
except BaseException:
    STATUS.write_text(json.dumps(dict(phase='failed', pid=os.getpid(), error=traceback.format_exc()), indent=2))
    raise
finally:
    save(notebook)
