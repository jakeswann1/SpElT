from pathlib import Path

import probeinterface as pi
import spikeinterface.extractors as se


def load_axona_ephys(path, probe_name):
    probe_path = Path(__file__) / "probes"
    recording = se.read_axona(path, all_annotations=True)
    if probe_name == "5x12_buz":
        probe = pi.read_prb(probe_path / "5x12-16_buz.prb").probes[0]
    else:
        print(f"Axona probe type not implemented in {__file__}")
    recording = recording.set_probe(probe)
