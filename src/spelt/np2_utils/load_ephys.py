from pathlib import Path
from typing import Literal

import spikeinterface.extractors as se


def load_np2_recording(path: Path, method: Literal["pcie", "onebox"]):
    if not path.exists():
        path = path.parent
    if method == "pcie":
        return se.read_openephys(path, stream_id="0", all_annotations=True)
    elif method == "onebox":
        return se.read_openephys(
            path, stream_name="Record Node 101#OneBox-100.ProbeA", all_annotations=False
        )
    else:
        raise NotImplementedError(
            f"Error: method {method} not implemented for loading NP2 recording"
        )


def load_np2_ttl(path: Path, recording_type: Literal["NP2_openephys", "NP2_onebox"]):
    if recording_type == "NP2_openephys":
        return se.read_openephys_event(path).get_event_times(
            channel_id="Neuropixels PXI Sync"
        )
    elif recording_type == "NP2_onebox":
        return se.read_openephys_event(path).get_event_times(
            channel_id="Neuropixels PXI Sync"
        )
