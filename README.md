# pyScan
### pyScan is a python-based toolkit for analysis of co-recorded electrophysiological and behavioural data, based on [scaNpix](https://github.com/LaurenzMuessig/scaNpix)
### Currently this is implemented for loading, processing and analysing raw data from Axona systems, with Neuropixels to be added soon
•	The package is based on Python 3.7, owing to several dependencies

•	Development by [Jake Swann](https://github.com/jakeswann1/) @ Wills-Cacucci Lab, UCL

# Usage
### This repo is designed to be used with the following workflow:
To extract and analyse CA1 pyramidal cell spike data:
1. Record raw axona files - _.bin_ and _.set_ files are required for each recording trial
2. Run _pyScan.preprocessing.loop_axona_spikeinterface.ipynb_
•	This runs collates individual trials into sessions, runs kilosort 2 and extracts position data
3. Manually curate spikes in phy
4. Run _pyScan.postprocessing.Check Cell Identity.ipynb_
•	This selects candidate pyramidal cells and saves their cluster IDs to a clusters_inc.npy
5. Run _pyScan.analysis.All Pyramidal Cells Analysis.ipynb_ for spike analysis
6. _pyScan.analysis.LFP analysis.ipynb_ (WIP) contains code to analyse LFP data from the same recordings
