# Optimization of Bivariate Multimodal Polynomials

This repository contains code for generating and optimizing bivariate multimodal polynomials. The goal is to compare the efficacy and efficiency of different optimization algorithms in reaching a global minima in bivariate multimodal functions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [License](#license)

## Introduction

Bivariate multimodal polynomials are polynomial functions with two variables that have multiple local minima and maxima. Optimizing such functions is a challenging task due to the presence of multiple local optima and the tendency of optimization algorithms to get stuck in these. This projects aimes to compare the efficacy and efficiency of Gradient Decent and Nelder-Mead optimization at reaching a global optima. The code contained in this repository can generate a set of multimodal polynomials by fitting functions to matrices filled with perlin noise. Then, Gradient Decent or Nelder-Mead optimization can be applied to these functions and data about the optimization is collected. 

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.