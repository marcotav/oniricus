## Marketing Mix Modeling [[view code]](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/pca/notebooks/pca-wines.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)

**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/unsupervised-learning/blob/master/pca/notebooks/pca-wines.ipynb) or by clicking on the [view code] link above.**


<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#goal"> Goal </a> •
  <a href="#imp"> Importing data </a> •
  <a href="#corr"> Correlations </a> •
  <a href="#fit"> Fitting a PCA </a> •
  <a href="#expvar"> Plotting the variance explained ratio of the PC </a> •
  <a href="#eig"> Component weights with corresponding variables for the PCs </a> 
</p> 


<a id = 'expvar'></a>
### 
### Author: [Marco Tavora](http://www.marcotavora.me/)

<a id = 'intro'></a>
### Introduction
import datetime  
import numpy as np  
import pandas as pd
import numpy.random as random
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels.formula.api as smf
import auxiliar_functions as af
pd.set_option('display.max_columns', None) 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" 
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
from scipy.stats import boxcox
from matplotlib import pyplot
from scipy import stats
import scipy
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
%matplotlib inline

## Table of contents

- [Introduction](#Introduction and Goal)
- [Goal](#Goal)
- [Data](#Data)
- [Functional Form](#Functional-Form)
- [Baseline](#Baseline)
- [Dynamics](#Dynamics)
- [Trend and seasonality](#Trend-and-seasonality)


<a id = 'Introduction'></a>
## Introduction and Goal
[[go back to the top]](#Table-of-contents)

In Marketing mix modeling (MMM) statistical models are used to:
- Understand the relationship between sales and the time series of different marketing channels 
- Forecast the impact of future marketing tactics
- Optimize advertising mix and promotional tactics with respect to the revue of ROI (return on investment)
The goal here is to build a model for the `sales` variable in our dataset

## Data

The data imported below contains:
- Weather variables (`sunshine`, `precipitation`, `temperature`)
- Competitors' expenditures
- Consumer Price Index `cpi`
- Impressions on different media channels 
- Event data (calendar)
- Average price of company's products

df = pd.read_csv('dataMMMv11.csv',index_col=0)
df.head()

### Type span of data

delta = pd.to_datetime(df['date'].tolist()[-1]) - pd.to_datetime(df['date'].tolist()[0])
print('Data has',delta.days,'days')

## How to incorporate digital media into MMM?

From Nielsen’s [Global AdView Pulse](http://www.nielsen.com/tz/en/insights/news/2014/the-billion-dollar-question-whats-the-real-return-in-investment-from-online-advertising.html):
> With click-through rates often below 1 percent, the study also found that clicks aren’t a good indication of incremental sales; delivered impressions are much more predictive of sales lift [...] [But] while impressions are important indicators of sales lift, some impressions are better than others. Total digital impressions — or total impressions by publisher — can provide overly coarse metrics, underestimating ROI. The more detailed impressions identified in the study (such as those by publisher, market, device and demographic) can give companies a clearer picture into their digital investments. 


We can try to use impressions but take into account:
- The impact of the impressions on company's typical consumers
- The level of awareness for a given impression
- Others

### Impressions distribution

The plots below show the number of impressions for Facebook and Search per day (in millions) for a period of two weeks.

two_weeks = df.iloc[175:190,:]
rcParams['figure.figsize'] = 10, 4
fig,ax = plt.subplots()
fig.suptitle('Facebook Impressions',fontsize=14)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Impressions (MM)',fontsize=14)
ax.bar(two_weeks['date'], two_weeks['impression_facebook']/1000000, width=0.7, color='b')
plt.xticks(rotation=45)

fig,ax = plt.subplots()
fig.suptitle('Display Impressions',fontsize=14)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Impressions (MM)',fontsize=14)
ax.bar(two_weeks['date'], two_weeks['impression_display']/1000000, width=0.7, color='g')
plt.xticks(rotation=45)


plt.show();

## Functional Form of $V_t$

The choice of the best possible functional form for the `sales` volume $V_t$ is crucial. Two standard options are *linear models* and *multiplicative models*. The former is the model we see everywhere. The latter is obtained by using $\ln V_t$ in place of $V_t$ and $\ln x_{it}$ in place of $x_{it}$. It is trivial to show that via a quick manipulation it acquires the multiplicative structure:

$$\ln {V_t} = \underbrace {\ln {\beta _0} + \sum\nolimits_{i = 1}^n {{\beta _i}} \ln {x_{it}} + \ln {\varepsilon _t}}_{\sum\nolimits_{i = 1}^n {\ln ({\beta _0}x_{it}^{{\beta _i}}{\varepsilon _t})} } \Rightarrow {V_t} = {\beta _0}\left( {\prod\nolimits_{i = 1}^n {x_{it}^{{\beta _i}}} } \right){\varepsilon _t}$$

the expression after the arrow is obtained by exponentiation. To compare both models we can use power transforms. From Wikipedia:
> In statistics, a power transform is a family of functions that are applied to create a monotonic transformation of data using power functions. This is a useful data transformation technique used to stabilize variance, make the data more normal distribution-like, improve the validity of measures of association such as the Pearson correlation between variables and for other data stabilization procedures.

We will use specifically the [Box-Cox transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation). The Box-Cox functional is a continuously varying function of the power parameter $\lambda$:

\begin{eqnarray}
h(V_t^{(\lambda )}) = \left\{ {\begin{array}{*{20}{l}}
{(V_t^\lambda  - 1)/\lambda }&{{\rm{if }}\lambda  \ne 0}\\
{\ln {V_t}}&{{\rm{if }}\lambda  = 0}
\end{array}} \right.
\end{eqnarray}

We then write:

$$h(V_t^{(\lambda )}) = \sum\nolimits_{i = 0}^n {{\beta _i}} h(x_{it}^{(\lambda )})$$

and determine $\lambda$ maximizing the maximum likelihood estimator (MLE) score. If $\lambda \approx 1$, $V_t^{(\lambda=1)}$ is close to linear 

$$V_t^{(\lambda )} = \sum\nolimits_{i = 0}^n {{\beta _i}} x_{it}^{(\lambda )}$$

whereas if $\lambda \approx 1$ the model is close top multiplicative:

$$V_t^{(\lambda )} = {\beta _0}\left( {\prod\nolimits_{i = 1}^n {x_{it}^{{\beta _i}}} } \right){\varepsilon _t}$$

The value of $\lambda$ that maximizes the [likelihood ratio](https://www.stat.berkeley.edu/~bartlett/courses/153-fall2010/lectures/11.pdf) is chosen. 

## Code for the Box-Cox transformation

### Converting to `datetime`

lst = pd.to_datetime(df.loc[:,'date']).tolist()
df.loc[:,'date'] = lst

sales = df[['date','sales']]
sales = pd.Series(sales['sales'].values, index=[d for d in sales['date']])

s = seasonal_decompose(sales)

tr = s.trend
tr.dropna(inplace=True)
dataframe = pd.DataFrame(tr.values)[:-1]
dataframe.columns = ['sales']
tr['sales'],lam = boxcox(dataframe['sales'])

print('lambda is:',round(lam,2))

## Trend, Seasonality and Residual

rcParams['figure.figsize'] = 12, 6
s = seasonal_decompose(sales)
fig = s.plot()
plt.show()

## Dynamics and Adstock

Advertising expenditures have impacts not only in the current time period, but usually for many periods. This lagged effect of advertising on consumer behavior is called **Adstock** (or "carry-over effect"). Suppose that sales volume $V_t$ can be modeled by:

$$ {V_t} = {\beta _0} + {\beta _1}f({x_{1t}}) + {\beta _2}f({x_{2t}}) + ...{\beta _n}f({x_{nt}}) + {\varepsilon _t}$$

where:

$$f({x_{it}}) = {x_{it}} + {\lambda _i}f({x_{it - 1}}),\,\,\,i = 1,...,n$$

This is equivalent to:

$$f({x_{it}}) = {x_{it}} + {\lambda _i}{x_{it - 1}} + \lambda _i^2x_{it - 2}^2 + ... + \lambda _i^nx_{it - n}^n$$

The adstock is obtained using `recursive_filter( )`. From the documentation using `n_coeff`$=n$:

      f[t] = ar_coeff[0] * f[t-1] + ar_coeff[1] * f[t-2]+ ... + ar_coeff[n-1] * f[t - n] + x[n]
      
Comparing with the above expression `ar_coeff[0]`$= {\lambda _i}$, `ar_coeff[1]`$= \lambda _i^2$, etc.

print(df.columns.tolist())

cols = ['impression_display', 'impression_facebook']

ar_coeff_1,ar_coeff_2 = 0.5, 0.5
display_adstock = tsa.filters.filtertools.recursive_filter(df['impression_display'], ar_coeff_1)
facebook_adstock = tsa.filters.filtertools.recursive_filter(df['impression_facebook'], ar_coeff_2)

display_adstock = af.vc_to_df(display_adstock,'display_adstock','')
df['display_adstock']= display_adstock
facebook_adstock = af.vc_to_df(facebook_adstock,'facebook_adstock','')
df['facebook_adstock']= facebook_adstock
df.head()

## Modeling using linear regression

Time series have a temporal linkage which can imply that one or both of these properties hold:
- Past values of the endogenous variable $y$ influence its current value 
- Errors $\varepsilon_t$ are correlated over time

If both properties are true, ordinary least squares (OLS) cannot be used to estimate its parameters. Let us suppose for the moment that we can use OLS for estimation of the parameters.


feature_names =  ['sunshine', 'precipitation', 'temperature', 'cpi', 'cci', 
                  'gdp', 'event', 'impression_display', 'impression_facebook', 'competition', 'price', 
                  'display_adstock', 'facebook_adstock']

cols = ['impression_display', 'impression_facebook','display_adstock', 'facebook_adstock']

for col in cols:
    df[col] = df[col]/1000000

df['competition'] = df['competition']/10000
df.head()

## Baseline Volume Calculation

The total volume can be defined in the following way:

$${V_t} = {V_{t,{\rm{baseline}}}} + \Delta {V_{t}}$$

where the second term is the incremental volume and is due to factors such as advertising, promotions, price reductions and so on. The baseline volume depends on factor such as regular shelf price, distribution, competitive activity, seasonality, etc

feature_names =  ['sunshine', 'precipitation', 'temperature','cpi', 'cci','gdp', 'event', 
                  'impression_display', 'impression_facebook','competition','price',
                  'display_adstock', 'facebook_adstock']

### Linear Model

Notice that since for now I will not do any forecasting (will not predict anything on yet-unseen data). I will [train and test with the full dataset](https://info.salford-systems.com/blog/bid/337783/Why-Data-Scientists-Split-Data-into-Train-and-Test).

X = df[feature_names].values
X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
y = df['sales']
model = sm.OLS(y, X)
model = model.fit();
model.summary()

## Analysis of the summary

The elements of the summary I will focus are:
- `DF Residuals`: Degrees of freedom of the residuals. 
- `F-statistic`: A measure how significant the fit is. The mean squared error of the model divided by the mean squared error of the residuals
- `Prob (F-statistic)`: The probability that you would get the above statistic, given the null hypothesis that they are unrelated
- `t`: The t-statistic value. This is a measure of how statistically significant the coefficient is.
- `P > |t|`: P-value that the null-hypothesis that the coefficient = 0 is true. If it is less than the confidence level, often 0.05, it indicates that there is a statistically significant relationship between the term and the response.
- `[0.025	0.975]`: The lower and upper values of the 95% confidence interval
- `Durbin-Watson`: A test for the **presence of autocorrelation** (that the errors are not independent.) Often important in time-series analysis
- `Cond. No`: A test for **multicollinearity** (if in a fit with multiple parameters, the parameters are related with each other).
- `AIC`: The Akaike Information Criterion. Adjusts the log-likelihood based on the number of observations and the complexity of the model.

Other elements are described in the appendix.

print('Parameters: ', [round(i,2) for i in model.params.tolist()],'\n')
print('Standard errors: ', [round(i,2) for i in model.bse.tolist()])

max_sales = max(list(df['sales']))  
min_sales = min(list(df['sales']))

> Confidence intervals around the predictions are built using the wls_prediction_std command

from statsmodels.sandbox.regression.predstd import wls_prediction_std
np.random.seed(9876789)

x = np.linspace(min_sales, max_sales, df.shape[0])
prstd, iv_l, iv_u = wls_prediction_std(model)

y_true = np.dot(X, model.params.tolist())

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, y_true, 'b', label="True")
ax.plot(x, y, 'o', label="data")
ax.plot(x, model.fittedvalues, 'r', label="OLS")
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size': 10});

print(df.columns.tolist())

cols = ['sunshine', 'precipitation', 'temperature', 'cpi', 'cci', 'gdp', 'event', 
        'impression_display', 'impression_facebook', 
        'competition', 'price', 'display_adstock', 'facebook_adstock']

## Testing and Validating Model

To test the null hypothesis $H_0$ that all ${\beta}_{i=1,n}$ are zero one uses:

$$F = \frac{{S{S_{{\rm{reg}}}}/n}}{{RSS/(T - K - 1)}}$$

where:

$$S{S_{{\rm{reg}}}} = \sum\limits_{i = 1}^n {{{({y_i} - {{\bar y}_i})}^2}} ,\,\,RSS = \sum\limits_{i = 1}^n {{{({y_i} - {{\hat y}_i})}^2}}$$

This test is done in the next cell and the $p$-value is essentially zero. From the [docs](http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html):

> This tests that each coefficient is jointly statistically significantly different from zero.

A = np.identity(len(model.params))[1:,:]
print(model.f_test(A))

## Error Analysis

> Hypothesis testing typically assumes that the noise values are mutually uncorrelated with zero mean and have the same Gaussian probability distribution – in other words, that the noise is white. If there is non-zero correlation between the noise values underlying different observations then the estimated model parameters are still unbiased, but estimates of their uncertainties (such as confidence intervals) will be biased (not accurate on average). This is also true if the noise is heteroskedastic – that is, if it has different variances for different data points.

Hence:

$$E({\varepsilon _t}) = 0,\,\,\,\,E(\varepsilon _t^2) = {\sigma ^2},\,\,\,\,E({\varepsilon _t}{\varepsilon _s}) = 0, {\varepsilon _t} \sim N(0,{\sigma ^2}),\,\,{{\bf{X}}^T}{\bf X}\,{\text{is non-singular}}$$

In the following we discuss how to deal with each possible violation.

### [Normality of residuals](http://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html)

#### Jarque-Bera test

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(model.resid)
lzip(name, test)

#### Omni test

name = ['Chi^2', 'Two-tail probability']
test = sms.omni_normtest(model.resid)
lzip(name, test)

max(model.resid_pearson)

residuals = model.resid_pearson
round(np.mean(residuals),2)
round(np.std(residuals)**2,2)

### [Durbin-Watson](http://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.durbin_watson.html)

> The null hypothesis of the test is that there is no serial correlation. The Durbin-Watson test statistics is defined as:

$$d = \frac{{\sum\limits_{t = 2}^T {{{({\varepsilon_t} - {\varepsilon_{t - 1}})}^2}} }}{{\sum\limits_{t = 1}^T {\varepsilon_t^2} }},$$

> For r = 0, indicating no serial correlation, the test statistic equals 2. This statistic will always be between 0 and 4. The closer to 0 the statistic, **the more evidence for positive serial correlation**. The closer to 4, the more evidence for negative serial correlation.

print('durbin_watson:',round(statsmodels.stats.stattools.durbin_watson(model.resid, axis=0),2))

plt.hist(model.resid,bins=25);
x = np.linspace(-2.5, 2.5, df.shape[0])
plt.ylabel('Count');
plt.xlabel('Residuals');

### $E(\varepsilon _t^2) = {\sigma ^2}$

> We should examine a plot of residuals versus fits. Obvious differences in the vertical spread of the residuals indicate nonconstant variance. The most typical pattern for nonconstant variance is a plot of residuals versus fits with a pattern that resembles a sideways cone.

resid = model.resid.tolist()
y_fitted = np.dot(X, model.params.tolist()).tolist()
rcParams['figure.figsize'] = 8, 4
fig,ax = plt.subplots()
fig.suptitle('Residues and fitted y',fontsize=14)
ax.scatter(y_fitted,resid,color='b')
plt.xticks(rotation=90);

### Multicollinearity

np.linalg.cond(model.model.exog)

> The ideal plot of residuals with each of the predictor should be a random scatter because we assume that the residuals are uncorrelated with the predictor variables. Any noticeable pattern in such plots indicates violation of linear relationship assumption.

resid = model.resid.tolist()
rcParams['figure.figsize'] = 4, 2

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[0]),fontsize=14)
ax.scatter(df[cols[0]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[1]),fontsize=14)
ax.scatter(df[cols[1]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[2]),fontsize=14)
ax.scatter(df[cols[2]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[3]),fontsize=14)
ax.scatter(df[cols[3]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[4]),fontsize=14)
ax.scatter(df[cols[4]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[5]),fontsize=14)
ax.scatter(df[cols[5]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[6]),fontsize=14)
ax.scatter(df[cols[6]], resid, color='b')
plt.xticks(rotation=90);


fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[7]),fontsize=14)
ax.scatter(df[cols[7]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[8]),fontsize=14)
ax.scatter(df[cols[8]], resid, color='b')
plt.xticks(rotation=90);

fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[9]),fontsize=14)
ax.scatter(df[cols[9]], resid, color='b')
plt.xticks(rotation=90);


fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[10]),fontsize=14)
ax.scatter(df[cols[10]], resid, color='b')
plt.xticks(rotation=90);


fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[11]),fontsize=14)
ax.scatter(df[cols[11]], resid, color='b')
plt.xticks(rotation=90);



fig,ax = plt.subplots()
fig.suptitle('Residues and {}'.format(cols[12]),fontsize=14)
ax.scatter(df[cols[12]], resid, color='b')
plt.xticks(rotation=90);


betas = af.vc_to_df(model.params,'betas','').iloc[:-1,:]
betas.index = ['sunshine', 'precipitation', 'temperature','cpi', 'cci','gdp', 'event', 
                  'impression_display', 'impression_facebook','competition','price',
                  'display_adstock', 'facebook_adstock']

betas

feature_names =  ['sunshine', 'precipitation', 'temperature','cpi', 'cci','gdp', 'event', 
                  'impression_display', 'impression_facebook','competition','price',
                  'display_adstock', 'facebook_adstock']

df1 = df.copy()
cols = ['impression_display', 'impression_facebook',
        'display_adstock', 'facebook_adstock']

df1[cols] = 0
df1.head()

X = df[feature_names].values
X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

X1 = df1[feature_names].values
X1 = np.concatenate([X1, np.ones((X1.shape[0], 1))], axis=1)

incremental = model.predict(X) - model.predict(X1) 

delta_V = af.lists_to_df(df1['date'].tolist(),
                         incremental.tolist(),'date','incremental')
print('Total Incremental:',round(delta_V['incremental'].sum(),2))

delta_V.head()

predictions = model.predict(X)
plt.figure(figsize=(10,5))
plt.scatter(predictions, y, s=30, c='b', zorder=40)
plt.scatter(predictions, predictions, s=30, c='r', zorder=40)
plt.xlabel("Predicted Values");
plt.ylabel("Actual Values");
plt.show()

## Elasticities

df_new = df.copy()
df_new['elasticity'] = np.log(df['sales']) - np.log(df['price'])
df_new.head()

df_new.plot(x='price', y='elasticity', style='o')

## Appendix

### Other elements of `statsmodels` summary

- Log-likelihood: The log of the likelihood function.
- BIC: The Bayesian Information Criterion. Similar to the AIC, but has a higher penalty for models with more parameters.
- std err: The basic standard error of the estimate of the coefficient. More sophisticated errors are also available.
- Skewness: A measure of the symmetry of the data about the mean. Normally-distributed errors should be symmetrically distributed about the mean (equal amounts above and below the line).
- Kurtosis: A measure of the shape of the distribution. Compares the amount of data close to the mean with those far away from the mean (in the tails).
- Omnibus D'Angostino's test: It provides a combined statistical test for the presence of skewness and kurtosis.
- Prob(Omnibus): The above statistic turned into a probability
- Jarque-Bera: A different test of the skewness and kurtosis
- Prob (JB): The above statistic turned into a probability
