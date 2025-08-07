Absolutely! Below is a **README.md** template in Markdown format, synthesized from your full technical documentation for both **USRP X310 bricked recovery** and your **WLAN interference study workflow**. You can copy-paste this directly into your repository as a well-organized, detailed README.

# USRP X310 Recovery & WLAN Interference Study

This document outlines how to recover a "bricked" Ettus USRP X310 device and offers a detailed workflow for conducting WLAN interference experiments using MATLAB and USRP hardware.

## Table of Contents

- [Chapter 1: Recovering a Bricked USRP X310](#chapter-1-recovering-a-bricked-usrp-x310)
  - [Common Reasons for Bricking](#common-reasons-for-bricking)
  - [Recovery Methods](#recovery-methods)
    - [Method 1: Using UHD Utilities](#method-1-using-uhd-utilities)
    - [Method 2: JTAG Programming via Xilinx Vivado Lab](#method-2-jtag-programming-via-xilinx-vivado-lab)
  - [Verification](#verification)
- [Chapter 2: WLAN Interference Study Project Structure](#chapter-2-wlan-interference-study-project-structure)
  - [Folder Organization](#folder-organization)
  - [Folder Descriptions](#folder-descriptions)
- [Chapter 3: Workflow for Performing Experiments](#chapter-3-workflow-for-performing-experiments)
- [Acknowledgments](#acknowledgments)

## Chapter 1: Recovering a Bricked USRP X310

Your USRP can become "bricked" due to errors such as firmware corruption, bad updates, power loss, or incorrect network settings.

### Common Reasons for Bricking

- **Firmware or FPGA image corruption**
- **Network misconfiguration (bad IP/subnet)**
- **EEPROM modification**
- **Power failures during firmware updates**
- **Overheating or hardware faults**
- **UHD/FPGA image version incompatibility**

### Recovery Methods

#### Method 1: Using UHD Utilities

1. **Prerequisites:**
   - Install [UHD (USRP Hardware Driver)](https://github.com/EttusResearch/uhd)
   - Add UHD `bin` folder to your PATH
   - Install [Xilinx Vivado Lab Edition](https://www.xilinx.com/support/download/index.html) (for FPGA image extraction)
   - Download the appropriate FPGA images for X310

2. **Steps:**
   - Check device connectivity:  
     `uhd_find_devices` (default IP: `192.168.10.2`)
   - Reload firmware:  
     `uhd_image_loader --args="type=x300,addr=192.168.10.2"`
   - Reflash FPGA:  
     `uhd_image_loader --args="type=x300,addr=192.168.10.2" --fpga-path `
   - Power cycle the USRP
   - Re-check connectivity:  
     `uhd_find_devices`

#### Method 2: JTAG Programming via Xilinx Vivado Lab

_Use only if Method 1 fails (e.g., severe FPGA failure)._

1. **Prerequisites:**
   - Install Vivado Lab Edition & Digilent Cable Driver
   - Download correct X310 FPGA `.bit` image

2. **Steps:**
   - Open Vivado Lab > Hardware Manager
   - `Tools` → `Auto Connect`
   - Right-click on detected FPGA, `Program Device`, select the appropriate image
   - Wait for programming to finish
   - Power cycle USRP

### Verification

- Run `uhd_find_devices`: check for device presence and correct firmware/FPGA info
- Run `uhd_usrp_probe`: test TX/RX functionality

If recovery fails, double-check image/version, connections, and try again.

## Chapter 2: WLAN Interference Study Project Structure

### Folder Organization

```
project-root/
│
├── Tx/           # Transmission scripts and waveform generators
├── Rx/           # Receiver scripts and analysis classes
├── Interference/ # Interference signal generators
└── Details/      # Experiment logs, configs, and related documentation
```

### Folder Descriptions

#### Tx/

- **usrptx.m**: Main transmission script for USRP B210
- **txwaveforms/**: Scripts to generate various WLAN signals:
  - `eht_waveform_gen.m`: 802.11be/EHT waveforms (SOI)
  - `vht_waveform_gen.m`: 802.11ac/VHT waveforms
  - `ht_waveform_gen.m`: 802.11n/HT waveforms
  - `nht_waveform_gen.m`: Non-HT
  - `hesu_waveform_gen.m`: HE SU (802.11ax)
  - `test_fmcw_int.m`: FMCW radar interference
  - `test_bt_int.m`: Bluetooth-like interference
  - `reference_signal.m`: Clean reference signal

- **Pilot Analysis Files:**  
  Scripts such as `ehtPilots.m` and `ehtPilotSubcarrierIndices.m` help define/modify pilot structure.

#### Rx/

- **WaveformAnalysisEngine.m**: MATLAB class for blind WLAN detection/analysis
- **recoverEHT.m**: Implements MIMO interference suppression
- **automate_analysis.m**: Batch-processing of RX data, EHT-MU filtering, visualization
- **mimoChannelEstimate.m**: Interference level estimation
- **wlanEHTDataBitRecover.m**: Enhanced LLR estimation
- **usrpRunrx.m**: RX/capture interface; file prompt, analysis/record modes
- **analyze_capture_data.m**: Captured data analysis—per-packet grouping, outlier removal, metrics (EVM, PEG, CPE, BER)
- **config.m**: Set RX time/BW/center freq/Gain/antennas

#### Interference/

- (Scripts for generating and configuring interference signals)

#### Details/

- Experiment descriptions, configs, and research papers.

## Chapter 3: Workflow for Performing Experiments

### 1. Select SOI & Interference

- Choose WLAN type (e.g., EHT, VHT) and interference type (WLAN, Bluetooth, FMCW)
- Generate waveforms as needed in `txwaveforms/`

### 2. Set Up the Transmitter

- Connect USRP to host PC
- Open `usrptx.m`, set # antennas, TX gain, duration, etc.
- Mount antennas; launch transmit

### 3. Configure the Receiver

- Open `config.m`: set RX time, BW, freq, gain, antennas
- Run `usrpRunrx.m` (choose capture or analysis)

### 4. Analyze Captured Data

- Use `analyze_capture_data.m` for metrics: EVM, SNR, CPE, PEG
- Repeat with varying interference TX gains for SINR sweeps

### 5. Interference Detection & Suppression

- Adjust script variables in Rx/ for suppression methods:
  - `1` = Frequency Excision
  - `2` = Adaptive (Kurtosis-based) Filtering
- Re-process, compare results

## Acknowledgments

- Ettus Research (UHD and USRP technical docs)
- MATLAB WLAN System Toolbox references
- Research community for interference detection/suppression methods
