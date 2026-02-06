from pathlib import Path

import numpy as np
import probeinterface as pi
import probeinterface.plotting as piplot
import spikeinterface.extractors as se


def generate_tetrodes(n):
    """
    Return a spikeinterface ProbeGroup object with n tetrodes.

    Tetrodes are spaced 300um apart vertically.
    """
    probegroup = pi.ProbeGroup()
    for i in range(n):
        tetrode = pi.generate_tetrode()
        tetrode.move([0, i * 300])
        tetrode.set_device_channel_indices(np.arange(i * 4, i * 4 + 4))
        probegroup.add_probe(tetrode)

    piplot.plot_probe_group(probegroup, with_channel_index=True)
    return probegroup


def get_probe(probe_name):
    """
    Load and return a probe configuration based on probe_name.

    Returns a tuple of (probe, num_channels).
    """
    probe_path = Path(__file__).parent / "probes"

    if "tetrode" in probe_name:
        num_channels = 32
        probe = generate_tetrodes(int(num_channels / 4))
    elif probe_name == "probe" or probe_name == "32 ch four shanks":
        probe = pi.read_prb(probe_path / "4x8_buzsaki_oneshank.prb")
        num_channels = 32
    elif probe_name == "5x12_buz":
        probe = pi.read_prb(probe_path / "5x12-16_buz.prb")
        num_channels = 64
    else:
        raise ValueError(
            f'Probe type "{probe_name}" not recognized. '
            'Please use "tetrode", "probe", "32 ch four shanks", or "5x12_buz"'
        )

    return probe, num_channels


def apply_probe_to_recording(recording, probe_name):
    """
    Apply probe configuration to a recording object.

    Slices the recording to the appropriate number of channels and sets the probe.

    Args:
        recording: SpikeInterface recording object
        probe_name: Name of the probe configuration to apply

    Returns:
        Modified recording object with probe set
    """
    probe, num_channels = get_probe(probe_name)

    # Cut to correct number of channels
    channel_ids = recording.get_channel_ids()
    recording = recording.channel_slice(channel_ids=channel_ids[:num_channels])

    # Set probe based on electrode type
    # For tetrodes, use ProbeGroup; for single probes, extract first probe from group
    if "tetrode" in probe_name:
        recording = recording.set_probegroup(probe, group_mode="by_probe")
    else:
        # For single probe types, extract the first probe from the ProbeGroup
        if hasattr(probe, "probes"):
            single_probe = pi.Probe.from_dict(probe.to_dict()["probes"][0])
            recording = recording.set_probe(single_probe)
        else:
            recording = recording.set_probe(probe)

    return recording


def load_axona_ephys(path, probe_name):
    """
    Load Axona ephys data and apply probe configuration.

    Args:
        path: Path to Axona data file
        probe_name: Name of the probe configuration to apply

    Returns:
        Recording object with probe set
    """
    recording = se.read_axona(path, all_annotations=True)
    recording = apply_probe_to_recording(recording, probe_name)
    return recording
