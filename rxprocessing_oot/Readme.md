# GNU Radio OOT Module: rxprocessing_ecological

This guide details the process of creating a custom Python-based Out-of-Tree (OOT) GNU Radio module named `rxprocessing_ecological`. This module will include two essential blocks: `matchedFilter` and `pktdetect`, and will be fully integrated into GNU Radio Companion (GRC).

---

## GNU Radio OOT Module Creation (Python Blocks): `matchedFilter` and `pktdetect`

---

### Goal

The objective is to create a GNU Radio **OOT module** called `rxprocessing_ecological`. This module will house two **Python blocks**: `matchedFilter`, which performs matched filtering via convolution, and `pktdetect`, designed for packet detection. Both blocks will be usable in **Python scripts** and **GNU Radio Companion (GRC)**.

---

### Prerequisites

Ensure the following are installed and configured:

* GNU Radio (version 3.10 or later recommended)
* Python 3 (version 3.8 or later)
* CMake, make, swig, and `gr_modtool`
* `PYTHONPATH` and `LD_LIBRARY_PATH` correctly set for custom installations

---

### 1. Create a New OOT Module

Begin by creating the module structure using `gr_modtool`:

```bash
gr_modtool newmod rxprocessing_ecological
cd gr-rxprocessing_ecological
````

This command establishes the necessary directory structure:

```
gr-rxprocessing_ecological/
  ├── python/
  ├── grc/
  ├── lib/
  ├── include/
  ├── CMakeLists.txt
  └── ...
```

-----

### 2\. Add the `matchedFilter` Python Block

Add the `matchedFilter` block using `gr_modtool`:

```bash
gr_modtool add -t general matchedFilter
```

Respond to the prompts as follows:

  * Block type: `general`
  * Language: `python`
  * Block name: `matchedFilter`
  * Arguments: `taps, threshold=0.0, verbose=False`

This action creates the file `python/rxprocessing_ecological/matchedFilter.py`.

-----

### 3\. Implement `matchedFilter.py`

Replace the content of `python/rxprocessing_ecological/matchedFilter.py` with the following:

```python
import numpy as np
from gnuradio import gr

class matchedFilter(gr.basic_block):
    def __init__(self, taps, threshold=0.0, verbose=False):
        gr.basic_block.__init__(self,
            name="matchedFilter",
            in_sig=[np.float32],
            out_sig=[np.float32])

        self.taps = np.array(taps, dtype=np.float32)[::-1]
        self.threshold = threshold
        self.verbose = verbose
        self.input_buffer = np.array([], dtype=np.float32)

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        self.input_buffer = np.concatenate((self.input_buffer, in0))

        if len(self.input_buffer) < len(self.taps):
            self.consume(0, len(in0))
            return 0

        result = np.convolve(self.input_buffer, self.taps, mode='valid')
        noutput_items = min(len(out), len(result))
        out[:noutput_items] = result[:noutput_items]
        self.input_buffer = self.input_buffer[noutput_items:]

        self.consume(0, len(in0))
        return noutput_items
```

-----

### 4\. Create `.block.yml` for `matchedFilter` in GRC

Create the file `grc/rxprocessing_ecological_matchedFilter.block.yml` with the following content:

```yaml
id: rxprocessing_ecological_matchedFilter
label: Matched Filter
category: '[rxprocessing_ecological]'

templates:
  imports: from gnuradio import rxprocessing_ecological
  make: rxprocessing_ecological.matchedFilter(${taps}, ${threshold}, ${verbose})

parameters:
- id: taps
  label: Filter Taps
  dtype: float_vector
  default: [1.0, -1.0, 1.0]

- id: threshold
  label: Detection Threshold
  dtype: float
  default: 0.0

- id: verbose
  label: Verbose
  dtype: bool
  default: false

inputs:
- label: In
  dtype: float

outputs:
- label: Out
  dtype: float

file_format: 1
```

-----

### 5\. Add the `pktdetect` Python Block

Add the `pktdetect` block using `gr_modtool`:

```bash
gr_modtool add -t general pktdetect
```

Respond to the prompts as follows:

  * Block type: `general`
  * Language: `python`
  * Arguments: `threshold=0.5, window_len=64, preamble=None, verbose=False`

This action creates the file `python/rxprocessing_ecological/pktdetect.py`.

-----

### 6\. Implement `pktdetect.py`

Replace the content of `python/rxprocessing_ecological/pktdetect.py` with the following:

```python
import numpy as np
from gnuradio import gr

class pktdetect(gr.basic_block):
    def __init__(self, threshold=0.5, window_len=64, preamble=None, verbose=False):
        gr.basic_block.__init__(self,
            name="pktdetect",
            in_sig=[np.float32],
            out_sig=[np.float32])

    def forecast(self, noutput_items, ninputs):
        return [noutput_items] * ninputs

    def general_work(self, input_items, output_items):
        ninput_items = min([len(items) for items in input_items])
        noutput_items = min(len(output_items[0]), ninput_items)
        output_items[0][:noutput_items] = input_items[0][:noutput_items]
        self.consume_each(noutput_items)
        return noutput_items
```

-----

### 7\. Create `.block.yml` for `pktdetect` in GRC

Create the file `grc/rxprocessing_ecological_pktdetect.block.yml` with the following content:

```yaml
id: rxprocessing_ecological_pktdetect
label: pktdetect
category: '[rxprocessing_ecological]'

templates:
  imports: from gnuradio import rxprocessing_ecological
  make: rxprocessing_ecological.pktdetect(${threshold}, ${window_len}, ${preamble}, ${verbose})

parameters:
- id: threshold
  label: Detection Threshold
  dtype: float
  default: 0.5

- id: window_len
  label: Correlation Window Length
  dtype: int
  default: 64

- id: preamble
  label: Preamble Sequence (comma-separated)
  dtype: float_vector
  default: [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

- id: verbose
  label: Verbose Output
  dtype: bool
  default: False

inputs:
- label: Float Input
  dtype: float

outputs:
- label: Matched Output
  dtype: float

file_format: 1
```

-----

### 8\. Update Python Bindings in CMake

It is crucial to ensure that both new Python block files are included in the CMake build process so they are properly installed and available for import. Edit `python/rxprocessing_ecological/CMakeLists.txt` to include both `matchedFilter.py` and `pktdetect.py`:

```cmake
gr_python_install(
    FILES
        __init__.py
        matchedFilter.py
        pktdetect.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/rxprocessing_ecological
)
```

-----

### 9\. Fix `__init__.py`

Edit or create `python/rxprocessing_ecological/__init__.py` to import both new blocks:

```python
from .matchedFilter import matchedFilter
from .pktdetect import pktdetect
__all__ = ['matchedFilter', 'pktdetect']
```

-----

### 10\. Build and Install

From your top-level module directory (`gr-rxprocessing_ecological`), execute the following commands to build and install the module:

```bash
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```

-----

### 11\. Test the Blocks in Python

You can verify the installation and basic functionality of the blocks in a Python script:

```python
from gnuradio import rxprocessing_ecological

# Test matchedFilter
taps = [1.0, 0.5, -0.5]
matched_filter_blk = rxprocessing_ecological.matchedFilter(taps)
print(matched_filter_blk)

# Test pktdetect
pktdetect_blk = rxprocessing_ecological.pktdetect(threshold=0.6)
print(pktdetect_blk)
```

-----

### 12\. Use in GNU Radio Companion

  * Launch GRC: `gnuradio-companion`
  * You will find both `Matched Filter` and `pktdetect` blocks listed under the category `[rxprocessing_ecological]` in the block sidebar.
  * Drag and drop them into your flowgraph, connect to appropriate float sources/sinks, configure their parameters (e.g., `taps` for `matchedFilter`, `threshold` for `pktdetect`), and run your flowgraph.

-----

### Optional Development Tips

During development, you can run directly from the source directory without full reinstallation each time. This is achieved by setting environment variables:

```bash
export PYTHONPATH=$PWD/python:$PYTHONPATH
export GRC_BLOCKS_PATH=$PWD/grc:$GRC_BLOCKS_PATH
```

-----

### To Uninstall

To remove the installed module files, execute the following commands:

```bash
sudo rm -rf /usr/local/lib/python*/dist-packages/gnuradio/rxprocessing_ecological
sudo rm -rf /usr/local/include/gnuradio/rxprocessing_ecological
sudo rm -f /usr/local/lib/libgnuradio-rxprocessing_ecological*
sudo rm -rf /usr/local/lib/cmake/gnuradio-rxprocessing_ecological*
sudo ldconfig
```

-----

### Summary of Steps

| Step | Purpose |
| :------------------ | :---------------------------- |
| `gr_modtool newmod` | Create OOT module structure |
| `gr_modtool add`    | Add individual Python blocks |
| Edit `.py` files    | Implement block logic         |
| Edit `CMakeLists.txt` | Register Python files for installation |
| Edit `__init__.py`  | Enable Python imports for blocks |
| Create `.block.yml` | Define GRC GUI representation |
| Build & Install     | Compile and install the module |
| GRC Usage           | Use blocks visually in GRC    |

```
```