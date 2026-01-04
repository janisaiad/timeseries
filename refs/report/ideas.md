comparing daily without scattering and with ; we can see without scattering improve stability and D1 has less asymetry

having asymetry very correlated with D1 for daily data is a factor of exogeneity which is very very common for daily data with breaking news, very few orderboook endogeneity in daily data, so we confirm this







with reproduce polands
added a function 'fit' if we choose a kernel to return the coefficients
rbf or linear kernel gives same results

3things manually doing


with PCA directions ,
3rd directions in the PCA, the trend 



they are less in the real part
they are not 






riding wavelets, data-mining fundamental knowledge in financial time series : 


# 1) introduction, paper main claim is that they 
- use Scattering spectra
- learn features confirmed to be related to financial time series analysis fundamental knowledge
- cojump analysis once they have learned features, cojumps arrving the same minute should all be asymetric (they confirm)

# 2) contributions
they don't have open sourced any thing 
our contribution, is implementing everything in a reproducible manner, that runs everywhere
and we use open source data (findable, accesible, interpretable, reusable)

structured of the code, timescale of the project involvment, work done by each members



# 3 method
wavelet features ; scattering features, PCA
jump detection (gumbel max-stable distribution threshold) ; r_t, sigma_t (we don't)
algorithmic methodology, we do ablation studies of SS, we change J, we change wavelet, threshold 
time series analysis methodology : we use different timescales, different filters (not handcrafted), we compare with asymetry(a knowledge that we bring)



# 3) data


data preprocessing : 
- 1st hour of the day and last hour removed
- OHLC data , close log returns, stack time series together
- how is r_t sigma_t computed  (EMA/high-low) jump score (show plot of log returns distribution threshold)

we show how many data : how many stocks, how many days, how many points, how many jumps



## 4 results

we should turn this as 
2) scattering spectra is not useful anymore 
1) we confirm something they have not done
3) timescales matters for engoneity / exogeneity 




plot SS/not

handcrafted/learnt

timescales/daily/hourly/5min

compare wavelet (D1  for several wavelets)



appendix 
we use different threshold 3 5 , we put threlshold plot , 
we compare J values / rbf for linear , svd decay plot







(cleaning the repo)
make 1 notebook to run


for jordi : 
changing wavelet (3 plots), svd plot, changing J values (3 plot D1/D2/D3) dot products, plots of D1
handcrafted filters

introduction/method



janis
use different thrshold for daily data (for 5 minutes already done)

scattering spectra is not useful anymore (metrics of that visually jumps )

complexity analysis of the code
plots for timescales matters for engoneity / exogeneity 


data preprocessing


analysis text : 
dataset description : deasasonalized vol is very stationary with some very big jumps as sstochastic volatility rough can give a prioir on it
r can be very correlated that's logical high r imply big vol, we can see volatility clustering that means it's non stationary (bar index 1500)
this means that dictionary learning method will fail, changepoint detection will not work with dict learning

for daily data it's mych noisy for sigma but it stabiliszes



12x5min = hourly, 12x hourly = daily
jump distirbutoin : we can see a gumbel fit well for each scales, with different threshold of interest
this is because of data quantity (exponentially less so threshold from 2.0 3.0 and 4.0 for daily, hourly, 5min) the fit is very goood
and very high extremes are not that much
(this leads to functional extreme pls to do PCA/PLS int he extremes where the gumbel don't fit to see if we still have that D1/D2/D3 in very high extremes)







feature distirbution : the feature distrib is not the same along kpca handcrafted and ss/noss especailly for D1
we cannot infer the same conclusions and the histogram do not behave the sme in log log plot


thresholds
when comparing thresholds (the most important int erms of statitical modelling) we see that it remains very stable
from 2.5 to 4.5 for 5min data we still have great stability, D1 profiles are the same, same for D2, 
but for D2 it degrades a bit

investigating we know that it's from having very high extremes that are not satisfying this mean-analysis of trend/following etc .. those are very much outliers




the time complexity is good, a bit long but it does not matter that much, 

goal of all this analysis is to make something invariant to regimes !! unless dictionary learning
