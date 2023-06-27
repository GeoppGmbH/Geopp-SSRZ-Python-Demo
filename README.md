[![Language](https://img.shields.io/badge/python-3.9%2B-yellow.svg?style=flat-square)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-AGPLv3+-red.svg?style=flat-square)](https://github.com/GeoppGmbH/Geopp-SSRZ-Python-Demo/blob/master/LICENSE)
[![Version](https://img.shields.io/badge/version-2.2-green.svg?style=flat-square)](https://github.com/GeoppGmbH/Geopp-SSRZ-Python-Demo/releases/tag/v2.2)
[![Reference](https://img.shields.io/badge/reference-SSRZ-blue.svg?style=flat-square)](https://www.geopp.de/SSRZ/)

## Geo++ SSRZ Python Demonstrator

Disclaimer
==================
The Geo++ SSRZ Python Demonstrator does not intend
to be a tool for post-processing or real-time positioning
applications. Therefore, it is not computationally optimized
for such operations.
Its purpose is to be a demonstrative tool
that accompanies the Geo++ SSRZ format documentation.

1. Introduction
   ============
   The Geo++ SSRZ Python Demonstrator is a software package
   designed to decode SSRZ binary files, 
   providing the SSRZ message content in a
   human-readable format and to compute the SSR components
   influence for a user location,
   dealing with general GNSS-related aspects.
   
   In summary, the Geo++ SSRZ Python Demonstrator
   can perform the following main tasks:
   - Decode SSRZ messages.
   - Compute a State Space Influence (SSI) for a user location
     (GNSS satellite coordinates are computed by using
	  broadcast navigation data). 
   
   The decoding involves the following messages:
    - Metadata (satellite, metadata, and grid groups)
    - Low and high-rate corrections
	- Global VTEC ionosphere corrections
	- Global satellite-dependent ionosphere corrections
    - Regional STEC corrections
	- Gridded ionosphere corrections
	- Regional troposphere corrections
	- Gridded troposphere corrections
	- QIX bias.
	
2. Requirements
   ============
   The Geo++ SSRZ Python Demonstrator has been developed
   and tested in Python 3.9 (64-bit) environment on Windows.
   
   The following elements are needed in order to use the demo:
   - a Python 3+ installation
	 (e.g., downloading Anaconda from https://www.anaconda.com/)
   - bitstruct Python module 
     (e.g., from cmd: "conda install bitstruct -c conda-forge")
   - crcmod Python module
	 (e.g., from cmd: "conda install crcmod -c conda-forge")  
   - tkinter Python module
      (e.g., from cmd: "conda install tk -c conda-forge")
   - a ssrz binary file (e.g. *.ssz).

   The python dependencies can be installed by using the following
   `pip` command in a command prompt.
   ```
   pip install -r requirements.txt
   ```
      
3. Notes
   =====
   Demo assumptions:
   - The number of regions is hard-coded to one.
   - The regional troposphere message ZM007 is always present and
     always the last one. Following this assumption,
	 the .csv output is filled with the SSI influence
	 at each epoch after the ZM007 message.
   
   The source code is formatted by adopting autopep8 
   (conformed to the PEP 8 style guide).

   The script "main.py" shows how to execute the demo.
   The required inputs are:
   - path of the ssrz binary file
   - name of the ssrz file
   - name of the RINEX navigation file
   - output folder (optional)
   - ellipsoidal (WGS84) latitude, longitude, and height
     (if no input coordinates are given default values are considered:
     lat = 52.5 deg, lon = 9.5 deg, height = 100 m)
   Other optional inputs are described in the source code.
   Remark: to decode, only the path and name of 
   the ssrz file are required.
   
   The script "start_ssrz_demo.py" is a simple GUI to execute the demo
   similarly to the "main.py" script.

   The results of the decoding of the input ssrz file (binary) 
   are saved in a text file named as the input file
   with "_ssr.txt" attached at the end of the name.
   
   The results of the ssr influence computation 
   are saved in a text file named as the input file
   with either "_ssi.txt" or "_ssi.csv" attached
   at the end of the name. The .csv file output is
   activated by setting csv_out variable True when calling
   the demo.
   
   In addition to the output files, a debug file is provided
   for the computation of the global VTEC.
   This file has the same name as the input file
   with "_gvi.txt" attached at the end.

   A test dataset is provided within the folder "test_data".
     
4. Reference
   ==========
   Geo++ State Space Representation Format (SSRZ)
   Document version 1.1.2
   https://www.geopp.de/SSRZ/

   
5. Additional information
   ======================
   Geo++ GmbH is the owner of the SSRZ Python Demonstrator.
   The position of one of the developers was initially funded 
   from the European Union's Horizon 2020
   research and innovation programme under the Marie Sklodowska-Curie
   Grant Agreement No 722023. 
   
