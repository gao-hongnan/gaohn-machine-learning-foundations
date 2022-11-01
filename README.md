## Workflow

### Installation

```bash
~/gaohn              $ git clone [https://github.com/gao-hongnan/gaohn-probability-stats.git gaohn_probability_stats](https://github.com/gao-hongnan/gaohn-machine-learning-foundations.git)
~/gaohn              $ cd gaohn-machine-learning-foundations
~/gaohn              $ python -m venv <venv_name> && <venv_name>\Scripts\activate 
~/gaohn  (venv_name) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn  (venv_name) $ pip install -r requirements.txt
~/gaohn  (venv_name) $ pip install myst-nb==0.16.0 
```

The reason for manual install of `myst-nb==0.16.0` is because it is not in sync with the current jupyterbook
version, I updated this feature to be able to show line numbers in code cells.

### Building the book

After cloning, you can edit the books source files located in the `content/` directory. 

You run

```bash
~/gaohn  (venv_name) $ jupyter-book build content/
```

to build the book, and

```bash
~/gaohn  (venv_name) $ jupyter-book clean content/
```

to clean the build files.

A fully-rendered HTML version of the book will be built in `content/_build/html/`.