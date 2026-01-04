# Riding wavelets

## 1-liner installation

```bash
chmod +x launch.sh && ./launch.sh
```

## Installation

This project is managed with `uv` (see `pyproject.toml` and `uv.lock`). The recommended way to install dependencies is `uv sync` (not `requirements.txt`).

1. Install uv:
   
   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. Using uv in this project:

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies (recommended):
   ```bash
   uv sync
   ```


   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
   ```bash
   uv pip list
   ```

## Warning

If `pip install uv` does not target the right Python, use `python3 -m pip install uv` instead.

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```

## What `launch.sh` does

`launch.sh` is a convenience bootstrap script that creates a local virtual environment in `.venv`, installs the project in editable mode, and runs a quick environment check.

You can run it from the repo root with:

```bash
bash launch.sh
```

Line-by-line:

- **`pip install uv`**: installs `uv` (a fast Python package/dependency manager).
- **`uv venv`**: creates a virtual environment in `.venv/`.
- **`source .venv/bin/activate`**: activates the environment for the current shell.
- **`uv pip install -e .`**: installs this repo as an editable package (so imports like `import model` / `import utils` resolve) and installs dependencies declared in `pyproject.toml`.
- **`uv cache prune`**: cleans `uv`'s download/build cache (optional; saves disk space, but may slow the next install).
- **`uv run tests/test_env.py`**: runs a small sanity-check script to verify the environment can import the local packages.
- **`source .venv/bin/activate`**: re-activates the venv (usually redundant; safe to remove if you want).


## Data 

The `data/stooq` directory contains datasets sourced from stooq.com, a free provider of financial market data. This folder typically holds historical and/or daily pricing data for various financial instruments, such as stocks, indices, currencies, and commodities. The contents are usually in the form of CSV or text files, where each file corresponds to a particular instrument or dataset fetched from stooq.

You can use the data in `data/stooq` for research, backtesting, or as sample financial market data for development and testing purposes. For full documentation or schema of the files, refer to the README inside `data/stooq` or see stooq.com's export documentation.

### Example: Sample stooq data file

Here is a snippet from  
`data/stooq/hungary/d_hu_txt/data/daily/hu/bse stocks/4ig.hu.txt`:

```
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
4IG.HU,D,20140613,000000,43.754,43.754,43.754,43.754,400,0
4IG.HU,D,20140626,000000,43.042,43.042,43.042,43.042,1850,0
4IG.HU,D,20140702,000000,42.116,42.116,41.46,41.46,12200,0
4IG.HU,D,20140703,000000,41.538,41.538,41.442,41.538,21800,0
...
```

- **Columns**:
  - `<TICKER>`: Symbol or code of the instrument
  - `<PER>`: Periodicity (`D` for daily)
  - `<DATE>`: Date (YYYYMMDD)
  - `<TIME>`: Time (usually `000000` for daily OHLC data)
  - `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`: Prices for the session
  - `<VOL>`: Volume
  - `<OPENINT>`: Open interest (often 0 for stocks)

For more information, see stooq.com's export format documentation or inspect the header row in your files.

