# Many-Particle-System-in-the-Plane
This repository is for use with Many-Particle-System-in-the-Plane project and includes all python files used for the simulation and experiments.

The project and all code were contributed by:
> Benoît Harwood, Isaac Ling, Zorawar Sidhu, George Taylor, Luke Alford

## Dependencies
|Package|Purpose|Installation|
|-------|-------|------------|
|`math`|General calculations throughout script|Build in python package|
|`time`|Measuring the time to run|Built in python package|
|`numpy`|Used for vector calculations and random function|Third party library|
|`numba`|Optimisation at runtime|Third party library|
|`matplotlib`|Plotting graphs|Third party library|
|`scipy`|An extension to numpy used for optimisation and constants|Third party library|

## Installation

Ensure your version of python is up to date or newer than
- Python 3.11.9


### Cloning from git

1. Clone the repository

```bash
git clone https://github.com/george-taylor03/Many-Particle-System-in-the-Plane.git
cd Many-Particle-System-in-the-Plane
```

2. Create and activate a virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Manual install of librarys

>[!NOTE]
> We recommend using an environment and installing dependencies using
> ```bash
> pip install -r requirements.txt
> ```

However the required libraries can be installed individually
```bash
pip install numpy
pip install numba
pip install matplotlib
pip install scipy
```

## Usage
