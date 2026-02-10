from pathlib import Path

import spikeinterface.extractors as se


def load_np2_pcie(path: Path):
    if not path.exists():
        path = path.parent
    return se.read_openephys(path, stream_id="0", all_annotations=True)


def load_np2_onebox(path: Path):
    if not path.exists():
        path = path.parent
    return se.read_openephys(
        path, stream_name="Record Node 101#OneBox-100.ProbeA", all_annotations=False
    )
