# Time-Series using ARMAX in Python
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)
## Introduction

A simple case of ARMAX process is:

&nbsp;&nbsp;&nbsp;&nbsp;
<img src="http://www.sciweavers.org/tex2img.php?eq=y_t%20%3D%20%7B%5Cbeta%7Dx_t%2B%20%5Cphi%20y_%7Bt-1%7D%2B%20%7B%5Cepsilon%7D_t%20&bc=White&fc=Black&im=png&fs=18&ff=mathdesign&edit=0" align="center" border="0" alt="y_t = {\beta}x_t+ \phi y_{t-1}+ {\epsilon}_t " width="229" height="27" />

where <img src="http://www.sciweavers.org/tex2img.php?eq=x_t&bc=White&fc=Black&im=png&fs=12&ff=mathdesign&edit=0" align="center" border="0" alt="x_t" width="20" height="14" /> is a covariate. For concreteness, let us consider the following data set:

<p align="center">
  <img src="datasetpic.png">
</p> 

