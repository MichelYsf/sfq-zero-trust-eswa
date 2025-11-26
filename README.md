# Security Friction Quotient (SFQ) – Zero Trust Identity Policy

This repository contains the code and configuration files used for the simulations
in the paper:

> **Security Friction Quotient for Zero Trust Identity Policy with Empirical Validation**

The goal of this project is to provide transparent and reproducible code so that
reviewers and other researchers can re-run the Monte Carlo simulations and
reproduce the figures and tables in the manuscript.

## Repository structure

- `src/`  
  Core source code for simulating identity activity and computing the SFQ.

- `data/`  
  Input parameters (e.g., policy definitions) and optionally example output files.

- `notebooks/` (optional)  
  Jupyter notebooks to regenerate figures and tables.

- `README.md`  
  This file.

## Requirements

- Python 3.9 or later
- The packages listed in `requirements.txt`

Install dependencies (example):

```bash
pip install -r requirements.txt
