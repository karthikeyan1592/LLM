import json
from pathlib import Path


class JsonlLogger:
    """
    Append-mode JSONL logger.  Each call to .log(dict) writes one JSON line.

    Usage:
        logger = JsonlLogger(path, overwrite=True)
        logger.log({"epoch": 1, "loss": 0.69})
        entries = logger.read_all()   # -> list of dicts
    """
    def __init__(self, path, overwrite=True):
        self.path = Path(path)
        if overwrite:
            self.path.unlink(missing_ok=True)

    def log(self, entry):
        with open(self.path, "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def read_all(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        p = tmp.name
    logger = JsonlLogger(p, overwrite=True)
    logger.log({"epoch": 1, "loss": 0.7})
    logger.log({"epoch": 2, "loss": 0.65})
    entries = logger.read_all()
    assert len(entries) == 2, f"FAIL: {len(entries)}"
    assert entries[1]["loss"] == 0.65
    os.unlink(p)
    print("PASS: JsonlLogger OK")
