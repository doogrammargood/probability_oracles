# Erdos Institute Miniprojects

This repository houses my Miniproject 1 and Miniproject 2 for the Erdos Institute, Fall 2026 for the Quantum Computing Bootcamp.

## Miniproject 1: Oracle Conversions

The goal of the Miniproject 1 is to convert from one quantum representation of information (a function oracle) to another (a phase oracle). See notebooks/prob_oracles.ipynb for more information.

## Miniproject 2: Durr-Hoyer Minimization

The goal of Miniproject 2 is to implement the Durr-Hoyer algorithm, which is an application of Grover's algorithm used for minimization. We use the algorithm to decode classical error correcting codes. See search.ipynb for more information

## In this Repository

- src (the main codebase):
    - oracle_objects (a folder of classes for our probabilty oracles)
    - linear_combination_of_unitaries.py
    - oracle_conversions.py
    - phase_estimation.py
    - search.py (main sourcecode for Miniproject 2)
- notebooks (for explaining the projects):
    - prob_orcales.ipynb:
        This Python notebook describes Miniproject1.
    - search.ipynb:
        This Python notebook describes Miniproject2.
- tests (for checking correctness):
    - test_oracles.py tests our implementation and conversion of several oracle types, and roughly corresponds to Miniproject 1.
    - test_search.py tests that we are correctly implementing Durr-Hoyer search and roughly corresponds to Miniproject 2.

The tests take a while to run. They take about 657 seconds on my laptop.