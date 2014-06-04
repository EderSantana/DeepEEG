#Joint Optimization of Algorithmic Suites for EEG Analysis
This code was used to generate the results of the paper at EMBC 2014.

##Prerequisites
1. numpy
2. scipy
3. scikit-learn
4. matplotlib
5. theano

They are all available for pip install. We also used the LogisticRegression class from LISA-Lab's [Deep Learning Tutorials](https://github.com/lisa-lab/DeepLearningTutorials).

##Data
The dataset used here was downloaded from http://www.bbci.de/competition/ii/ .

##Running the experiments
We suggesting running the experiments at an IPython section:  
`run piecewise_deep_csp`  
To visualize the temporal projection, run  
`plot(avg_v.get_value())`

##Plataform
We run the experiments reported on the paper on a MacBook Air OS 10.8 with
1. numpy 1.6
2. scipy 0.13
3. scikit-lern 0.13
4. We had no user defined theano flags. Most importantly, we this means that the experiments run on CPU with float64 precision.
