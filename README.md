# Reversing Game of Life

This repository contains our approach of solving the [challenge](https://www.kaggle.com/c/conway-s-reverse-game-of-life) posted on Kaggle back in the days.
The whole algorithm is described on [Medium](https://medium.com/@ptyshevs/rgol-ga-1cafc67db6c7) in great detail. You can find accompanying code in `ga_tutorial.ipynb`.

Almost equivalent code is used in `GeneticSolver`, with multiprocessing version of fitness scoring added. `MPGeneticSolver` goes one step further by running multiple `GeneticSolver`'s in parallel and returning the best scoring solution across all of the cores. Cythonized version of `make_move` is written in `life.pyx`, which is responsible for ~8x speedup in fitness scoring, and thus the overall performance.

## Results

Genetic Algorithm have yielded `0.0634` score on private test set, which beats top solutions by a wide margin.
Another approaches were tried, including Random Forest and Neural Networks on sliding windows, with top score ~`0.12034`.

## Team

* Andrew Stadnik ([@anstadnik](https://github.com/anstadnik))
* Vlad Paladii ([@samaelxxi](https://github.com/samaelxxi))
* Pavel Tyshevskyi ([@ptyshevs](https://github.com/ptyshevs))
