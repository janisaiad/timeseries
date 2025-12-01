# Project Name

## Description
A clear and concise description of what this project does and what it is for.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Tests](#tests)
- [License](#license)
- [Contact](#contact)
## Installation

To install dependencies using uv, follow these steps:

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

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
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

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of ```launch.sh```

Replace with your project folder name (which means the name of the library you are deving) in :```tests/test_env.py: ```


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

