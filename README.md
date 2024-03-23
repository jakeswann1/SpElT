
![spelt logo med](https://github.com/jakeswann1/SpElT/assets/66915197/a9c365a9-fecb-42fa-aae5-62f2042b0c81)
# Spelt - the Spatial Electrophysiology Toolbox
### Spelt is a python-based toolkit for analysis of co-recorded electrophysiological and behavioural data, based on [scaNpix](https://github.com/LaurenzMuessig/scaNpix), and using [SpikeInterface](https://github.com/SpikeInterface/spikeinterface)
### Currently this is implemented for loading, processing and analysing raw data from Axona systems, with Neuropixels 2 Open Ephys recordings as a WIP
•	The package code requires Python 3.8, owing to several dependencies

•	Development by [Jake Swann](https://github.com/jakeswann1/) @ Wills-Cacucci Lab, UCL

# Usage
### This repo is currently designed to be used with the following workflow:
To extract and analyse CA1 pyramidal cell spike data:
1. Record raw Axona or OpenEphys files
2. Run _Spike Sorting From Sheet.ipynb_ (requires correctly formatted Google Sheet and data directory, or you can load your own .xlsx)
•	This runs collates individual trials into sessions, runs Kilosort 2/4 (depending on probe type) and extracts position data
3. Manually curate spikes in phy
4. Run _pyScan.postprocessing.Check Cell Identity.ipynb_
•	This selects candidate pyramidal cells and saves their cluster IDs to a clusters_inc.npy
5. Run _pyScan.analysis.All Pyramidal Cells Analysis.ipynb_ for spike analysis
6. _pyScan.analysis.LFP analysis.ipynb_ (WIP) contains code to analyse LFP data from the same recordings
