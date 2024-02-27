## Installation
First, clone the repository
```
git clone ...
cd jaxued
```

Install:
```
pip install -e .
```

Follow instructions [here](https://jax.readthedocs.io/en/latest/installation.html) for jax GPU installation, and run something like the following 
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


And run:
```
python examples/minigid_plr.py
```

## Running
We provide three example files, `examples/maze_{dr,plr,paired}.py` implementing DR, PLR (and ACCEL) & PAIRED, respectively.
Each of these is standalone implementations of each of these algorithms

## Tweaking
To start modifying how an algorithm works, or to add a new environment, simply copy-paste the example code and start making changes!
