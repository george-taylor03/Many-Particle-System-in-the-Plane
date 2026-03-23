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

### Manual install of libraries

>[!NOTE]
> We recommend using an environment and installing dependencies using
> ```bash
> pip install -r requirements.txt
> ```
> [See above](https://github.com/george-taylor03/Many-Particle-System-in-the-Plane?tab=readme-ov-file#cloning-from-git)

However the required libraries can be installed individually
```bash
pip install numpy
pip install numba
pip install matplotlib
pip install scipy
```

## Usage
For Task 1 see `SimulationStep.py`

For Task 2 see `Fluctuations.py`

For Task 3 see `runCompression.py`

For Task 4 see `Gravity.py`

### `SimulationStep.py`

Contains the function SimulationStep that preforms one single step forward in time.

- Args:
    - dt (float): Time step size
    - x (2xN array): Current particle positions (first row are horizontal coordinates, second row are vertical coordinates)
    - v (2xN array): Current particle velocities (first row are horizontal coordinates, second row are vertical coordinates)
    - part (dictionary['radius','spring']): Stores properties of the particle (radius r of all particles and spring constant of all particles)
    - box (2x2 array): Array storing the corners of the box (np.vstack([low,upp]))
    - g (float): Single number, the amount g of gravity

- Returns:
    - tuple: xnew(2xN array), vnew(2xN array) which are the updated position and velocity

> [!IMPORTANT]
> This file does nothing when run. The function needs to be imported into other files.

### `Fluctuations.py`
This file is for Task 2 (T2)

At the start there are 3 variables that can switched from `True` to `False` depending on which part of the question you want to run. All can be `True` at the same time.

|Part|Usage|
|----|-----|
|`partA`|Plots a loglog plot of number of particles against standard deviation of temperature|
|`partB`|Plots a histogram of the distribution of particle speeds|
|`partC`|Prints the value of the mean free path|

Run using:
```bash
python ./Fluctuations.py
```
### `runCompression.py`

This file is for Task 3 (T3)

Shows the compression animation live

> [!CAUTION]
> The animation takes a couple of minutes to run

Prints:
- Initial average pressure
- Compressed average pressure
- Compressed average temperature

Plots temperature over time for the predetermined time frame

Plots average pressure over time for the same predetermined time frame

### `Gravity.py`