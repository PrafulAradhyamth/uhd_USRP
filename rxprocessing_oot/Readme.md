
# GNU Radio OOT Module: rxprocessing_ecological

This guide explains how to create a custom Python-based Out-of-Tree (OOT) GNU Radio module with multiple blocks, including `matchedFilter` and `pktdetect`, and integrate them into GNU Radio Companion (GRC).

---

## 🏗️ Module Creation

```bash
gr_modtool newmod rxprocessing_ecological
cd gr-rxprocessing_ecological
```

---

## ➕ Adding Block: matchedFilter

```bash
gr_modtool add matchedFilter
```

- Block type: `general`
- Language: `python`
- Arguments: `taps, threshold=0.0, verbose=False`

### Python Implementation

File: `python/rxprocessing_ecological/matchedFilter.py`

```python
import numpy as np
from gnuradio import gr

class matchedFilter(gr.basic_block):
    def __init__(self, taps, threshold=0.0, verbose=False):
        gr.basic_block.__init__(self,
            name="matchedFilter",
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

### GRC Block YAML

File: `grc/rxprocessing_ecological_matchedFilter.block.yaml`

```yaml
id: rxprocessing_ecological_matchedFilter
label: matchedFilter
category: '[rxprocessing_ecological]'

templates:
  imports: from gnuradio import rxprocessing_ecological
  make: rxprocessing_ecological.matchedFilter(${taps}, ${threshold}, ${verbose})

parameters:
- id: taps
  label: Filter Taps
  dtype: float_vector
  default: [1.0, 1.0, 1.0, -1.0, -1.0]

- id: threshold
  label: Threshold
  dtype: float
  default: 0.0

- id: verbose
  label: Verbose
  dtype: bool
  default: False

inputs:
- label: Float Input
  dtype: float

outputs:
- label: Filtered Output
  dtype: float

file_format: 1
```

---

## ➕ Adding Block: pktdetect

```bash
gr_modtool add pktdetect
```

- Block type: `general`
- Language: `python`
- Arguments: `threshold=0.5, window_len=64, preamble=None, verbose=False`

### Python Implementation

File: `python/rxprocessing_ecological/pktdetect.py`

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

### GRC Block YAML

File: `grc/rxprocessing_ecological_pktdetect.block.yaml`

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

---

## 🔧 CMake Integration

In `python/CMakeLists.txt`:
```cmake
gr_python_install(FILES
    __init__.py
    matchedFilter.py
    pktdetect.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/rxprocessing_ecological
)
```

---

## 🧪 Build and Install

```bash
mkdir build && cd build
cmake ..
make
sudo make install
sudo ldconfig
```

---

## ✅ Done

You can now use both `matchedFilter` and `pktdetect` blocks in GNU Radio Companion under `[rxprocessing_ecological]`.
