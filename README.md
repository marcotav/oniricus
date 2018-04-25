# Time-Series using ARMAX in Python
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)
<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#data"> Data </a> •
</p>

<a id = 'intro'></a>
## Introduction

A simple case of ARMAX process is:

&nbsp;&nbsp;&nbsp;&nbsp;
<p align="center">
  <img src="ARMAX.png" 
       width="200">
</p>
where x is a covariate. Note that the covariate coefficient is not straightforward to interpret. Its value is not, as in the case of linear regressions, the effect of x on y when the former increases by one unit, but its interpretation depends on the previous values of the response y.

<a id = 'data'></a>
## Data

For concreteness, let us consider the following data set:

<p align="center">
  <img src="datasetpic.png", width="300",height="300">
</p> 

