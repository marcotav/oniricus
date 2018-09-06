# Option Pricing with Cointegrated Assets

### Notebook by [Marco Tavora](https://marcotavora.me/)

## Table of contents

1. [Introduction](#Introduction)

## Introduction
[[go back to the top]](#Table-of-contents)

A simple model for option hedging based on cointegrated vectors is:

<br>
<br>
<p align="center">
  <img src="images/modeloptions.png" 
       width="400">
</p>
<br>

The inclusion of a *stationary* process z presupposes that in the long-run, stationarity is eventually achieved. The volatility is a simple GARCH(1,1) process. To control for growth and violation of mean reverting behavior, we added a deterministic term to z process.

## Conceptual Bird's Eye Review
[[go back to the top]](#Table-of-contents)

### Vector Error Correction Model VECM

The VECM model is given by:

<br>
<p align="center">
  <img src="images/VECM.png" 
       width="900">
</p>




where we ${y_{1,t}}$ is the SPY and ${y_{1,t}}$ is the SHY. If both series are cointegrated, this information is included in the model via the error correction terms:

$${\alpha _i}({\beta _0}\, + {\beta _1}{y_{1,t - 1}}\, + {\beta _2}{y_{2,t - 1}}),\,\,\,\,\,i = 1,2$$

where 

$$({\beta _0}\, + {\beta _1}{y_{1,t - 1}}\, + {\beta _2}{y_{2,t - 1}}),\,\,\,\,\,i = 1,2$$

corresponds to the long-run. If in the long-run the last equation is zero we have the folowing relation:

$$\,{y_{2,t - 1}} =  - ({\beta _0}/{\beta _2})\, - ({\beta _1}/{\beta _2}){y_{1,t - 1}}$$

Hence, the $\beta$s we obtain after fitting the VECM inform us about the equilibrium relationship between the time series. When the two series deviate from equilibrium, the $\alpha$s "push them back".

We will choose $y_1$ and $y_2$ to be the following time series:
- SPY (the S&P 500 exchange traded fund)
- SHY (iShares 1-3 year Treasury Bond) prices. 

# install.packages('ggplot2')
# install.packages('xts')
# install.packages('quantmod')
# install.packages('broom')
# install.packages('tseries')
# install.packages("kableExtra")
# install.packages("knitr")
# install.packages("vars")
# install.packages("urca")

To load the data we will use the library `quantmod` which contains the function `getSymbols`. 

From the [documents](https://www.rdocumentation.org/packages/quantmod/versions/0.4-13/topics/getSymbols)

> getSymbols is a wrapper to load data from various sources, local or remote.

In our case we will load data from Yahoo Finance.

rm(list=ls()) 
library(tseries)
library(dynlm)
library(vars)
library(nlWaldTest) 
library(lmtest)
library(broom) 
library(car)
library(sandwich)
library(knitr)
library(forecast) 

library(quantmod)
setSymbolLookup(SHY='yahoo',SPY='yahoo')
getSymbols(c('SHY','SPY'))  

Defining `y1` and `y2` as the adjusted prices and joining them:

y1 <- SPY$SPY.Adjusted
y2 <- SHY$SHY.Adjusted
time_series <- cbind(y1, y2)
print('Our dataframe is:')
colnames(time_series) <- c('SHY', 'SPY') 
head(time_series)

We can check for stationary of the series individually:

adf.test(time_series$SHY)
adf.test(time_series$SPY)

print('p-values:')
adf.test(time_series$SHY)$p.value
adf.test(time_series$SPY)$p.value

par(mfrow = c(2,1))
plot.ts(time_series$SHY,
        type='l')
plot.ts(time_series$SPY, 
        type='l')

### Johansen Test for Cointegration

The most well known cointegration test is the Johansen test which estimates the VECM parameters and determines whether the determinant of 

$$\alpha \beta = \left[ {\begin{array}{*{20}{c}}
{{\alpha _1}}\\
{{\alpha _2}}
\end{array}} \right]\left[ {{\beta _0}\,\,\,{\beta _1}\,\,\,{\beta _2}} \right]$$


is zero or not. If $\alpha \beta\neq 0$, the series are cointegrated.

library(urca)

johansentest <- ca.jo(time_series, type = "trace", ecdet = "const", K = 3)
summary(johansentest)

The lines $r=0$ and $r\le 1$ are the results of the test. More specifically:

- line $r=0$: these are the results of the hypothesis test with null hypothesis $r=0$. More concretely, this test checks if the matrix has zero rank. In the present case the hypothesis is rejected since the test variable is well above the $1\%$ value;
- line $r\le 1$: these are the results of the hypothesis test $r\le 1$. Now since the test value is below the $1\%$ value value we fail to reject the null hypothesis. Hence we conclude that the rank of $\alpha \beta$ is 1 and therefore the two series are cointegrated and we can use the VECM model.

Note that if hypotheses were reject we would have $r=2$ corresponding to two stationary series.

### Fitting the VECM

t <- cajorls(johansentest, r = 1)
t

The $\beta$ coefficients are given above. The $\gamma$ coefficients are above the $beta$s in the output.

## TO BE CONTINUED

