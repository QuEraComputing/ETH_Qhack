General directions on the challenge statement can be found in the _YQuantum_2025_challenge.pdf_ file. Follow below for extra relevant information.

## Installing Bloqade and other packages 

All work for this challenge will be based on Bloqade, QuEra's neutral atom emulator and SDK, and, if you feel insispired for something a bit more advanced, on Kiring, our compiler development infrastructure.

Installation instructions for Bloqade and all sub-packages necessary can be found here [here](https://bloqade.quera.com/latest/), and amount to little more than `pip install bloqade`. The little more is the Bloqade implementation of the PyQrack simulator, which can be obtained via `pip install 'bloqade-pyqrack[pyqrack]'`.

## Documentation

This year’s YQuantum challenges require a _notebook_ implementation that is heavily considered during judging, as well as generating a small slide presentation to explain your experience and development to the judges. Your notebook with all circuit solutions should be executable, so be sure to include a virtual environment with all the needs.

### Submissions and pre-submissions

For that, here is the guideline:
1. Fork this repo, place all the code you wrote in one folder with your team name under the `team_solutions/` folder (for example `team_solutions/quantum_team`).
2. Create a new entry in `team_solutions.md` following the format shown that links to the folder with your solution and your documentation.
3. Create a Pull Request from your repository to this original challenge repository
4. Submit the "challenge submission" form

### Evaluationm criteria