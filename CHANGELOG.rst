Changelog
=========

Current (2025-07-10)
--------------------

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
