
we worked 50+30hrs together on the project

we coded and deployed 2k LOC for backend automated computations, and open source 
(write structure of code as a graph with utils/ , model/ and notebooks/ simple structure etc .. to see where each calculation is performed)

we do not work in finance and we are just curious of having very noisy data with highly non trivial spectral analysis/dictionary learning

we took a paper from a hedge fund that is interesting, has fundamental knowledge, use method that is non stationary by nature, 
we re implemented the whole without any prior on the code, just methodology from the paper
and combined this with fully open source and reproducible data, everything is reproducible correspond to the highest standards in terms,
even the jupyter files are combined with python files and for maximum reproducibility, all of this with a 1 liner fast installation,
so this is the highest standards in terms of reproducibility


the time complexity of our software is as fast as possible without parallel computation for wavelet convolution


for the data, we found the data but the format is not ideal, fat and OHLC and there was different timescales so we tested them all, in terms of methodology
this is also a matter of testing the most important know fact in financial markets 'they are scale invariant' so this is our 1st research question, 
can we see this invariance in our data mining ?
so we produced the data handling method to transform raw data into a single log return vector of jump scores, that is our quantity of interest to analyze


1st our data should meet the hypothesis to deliver the theoretical results, that means the qq plot with gumbel of jump distrib should be 
- we can see that gumbel distrib is well fitted, extremes are not that much well fitted due to the gaussian hypothesis being strong for extremes and heavy tail
distribution in financial returns (that we know is empirically true)
- for 5 minutes the gumbel deviates a bit, this shows that a frechet distribution would be a better fit in the extremes but under the frechet hypothesis 
the threshold is different, which in fact do not matter that much for PCA and mean analysis (if we wanted to estimate variance and extrapolate understanding of spectral decay
this should matter), it's just the fact to be max-stable distribution that is important in practice so that we can obtain a consistent and stationary threshold
(the hypothesis is more about jump score being stationary than being gumbel), for gumbel+frechet we should just calculate threshold quantile differently


so the data satisfy core hypothesis to give a sense of kernel PCA after threshold




our research question leads us to testing several hypothesis : 
- we should see consistent jump profiles along thresholds
- we should see less implications of scattering spectra due to volatility already encoded into the jump score, because under the gaussian hypothesis we don't have volatility clustering and hence vol of vol ie vol of jump score is unuseful

- we should see consistent profiles along timescales and along markets (poland, hongkong, hungary, us, uk)

- higher scales means more exogeneity and macro, so more reflexivity for 5min, more asymetry for 


spoiler this is all we see (dispatch each question and answers with plots with 1 slide for each question)



at the end the final goal in the paper is to analyze and justify cojump analysis under exo/endo via the reflexivity (done in cojump.py, reserve a plot i'll put it)


'Hence, themost strikingconclusionof this section is
thatmany large co-jumpsare in fact explainedbyen
dogenousdynamicsandpropagateacrossstocks" 



for D1 with different wavelets we have consistent results, across very different ones (show their time/frequency profile in tikz) (fbsp, cmor, cgau)

for the asymetry we confirm the finding that asymetry is very correlated for daily data, and uncorrelated for 5 min one so we confirm also financial


for D3 in general it is unstable for poland, hungary,  because trend is very hard to state we need more data ;; (put placeholder for D3 for US data)