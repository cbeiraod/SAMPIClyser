Changelog
=========

Current (2025-10-15)
--------------------

0.1.1 (2025-10-15)
------------------

* Added an empty root data generator for setting dtypes correctly
* Modularised the schema code for processing data in a more unified and central way
* Added get_file_metadata to the default module imports
* Added CLI for checking time ordering
* Added function to determine if hits are time-ordered or not
* Added batch_size option to most of the sampiclyser utilities


0.1.0 (2024-09-12)
------------------

* Fixed some bugs from previous release


0.0.8 (2024-09-12)
------------------

* Added a function for steering the waveform plotting
* Made previous plotting functions respect the selected style
* Added a function for setting style for sampiclyser plotting
* Added a function to handle title and labels for waveform plots
* Added a utility function to convert numbers to their corresponding ordinal
* Added CLI interface for hitmap plotting
* Added a utility to generate an example config file for the hitmap generation
* Added a new entry point for the sampiclyser utilities, everything that is not directly related to analysing the data but required for it
* Fixed hints in the CLI


0.0.7 (2024-07-24)
------------------

* Added helper function to add finishing touches to the legends of waveform plots
* Added helper function to plot single waveforms
* Added function to reorganize the samples from the circular buffer
* Added function to yield selected waveforms based on channel and sequential hit position
* Finished refactoring hit reading
* Small fix to rate plots so 0 bins are shown
* Added entry point only for printing channel hits
* Started refactoring SAMPIC hit reading into its own dedicated function, yielding batches
* Added steering function to apply interpolation
* Added Lanczos interpolation function
* Added windowed sinc interpolation function
* Added scipy as a dependency for signal processing
* Added function to parse file metadata and extract the sampling frequecy
* Added function and command line command to plot hit rate vs time for specific SAMPIC channels
* Converted command line interface to click


0.0.6 (2024-06-27)
------------------

* Added function for plotting hit rate over time


0.0.5 (2024-06-27)
------------------

* Fixed get_channel_hits batch processing of feather files
* Added docstrings to all functions/methods
* Added functions for dealing with header metadata from files


0.0.4 (2024-06-26)
------------------

* Fixed metadata storing
* Fixed metadata stored in bytes so that actualy data is actually stored


0.0.3 (2024-06-19)
------------------

* Added script with an entrypoint for running the conversion tool from the command line
* Added hit calculation on root files
* Added header as metadata for feather and parquet files
* Added header as metadata for root files
* Added example sensor specs to the sensor hitmaps for future documentation


0.0.2 (2024-06-19)
------------------

* Added SAMPIC binary decoding
* Added some tools to handle hits and sensor hitmaps
* First code release on PyPI.


0.0.1 (2024-06-19)
------------------

* Test empty release
