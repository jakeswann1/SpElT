import spikeinterface as si
from probeinterface import Probe


def get_depth_channels(
    recording: si.BaseRecording | si.SortingAnalyzer,
) -> tuple[list[int], list[float]]:
    """
    Identifies channels belonging to the depth shank and their y positions.

    Parameters
    ----------
    recording : si.BaseRecording
        The input recording with multiple shanks.

    Returns
    -------
    tuple
        Tuple containing:
        - List of channel indices belonging to the depth shank
        - List of corresponding y positions (depths) for those channels
    """
    probe: Probe = recording.get_probe()

    # Select all channels on the column with the greatest total y extent
    unique_x_positions = set(probe.contact_positions[:, 0])
    max_extent = 0
    depth_shank_channels = []
    for x in unique_x_positions:
        channels_on_column = [
            i for i, pos in enumerate(probe.contact_positions) if pos[0] == x
        ]
        y_coords = probe.contact_positions[channels_on_column, 1]
        extent = y_coords.max() - y_coords.min()
        if extent > max_extent:
            max_extent = extent
            depth_shank_channels = channels_on_column

    # Sort channels by y position and get corresponding depths
    depth_shank_channels.sort(
        key=lambda ch: probe.contact_positions[ch, 1],
        reverse=True,  # Change to False if you want shallow-to-deep
    )

    depth_positions = [probe.contact_positions[ch, 1] for ch in depth_shank_channels]

    return depth_shank_channels, depth_positions
