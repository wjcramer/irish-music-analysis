# Traditional Irish Music: Tune Structure & Mode Analysis

This project analyzes the structure and modal properties of traditional Irish tunes using data from [TheSession.org](https://thesession.org). It scrapes tune metadata and ABC notation to study patterns in tune types, meters, keys, and modes.

## Features
- Scrapes tune metadata and ABC from TheSession.org
- Parses tunes using `music21`
- Analyzes modes and structural patterns

## Setup
```bash
git clone https://github.com/yourusername/irish-music-analysis.git
cd irish-music-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt