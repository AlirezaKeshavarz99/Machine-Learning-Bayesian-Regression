# Machine-Learning-Bayesian-Regression
Machine Learning Project - Using Bayesian Regression to Predict Bitcoin Price
Predicting Bitcoin Price Variations using Bayesian Regression

Following are the steps which were taken in this project:

1. Compute the price variations (Δp1, Δp2, and Δp3) for train2 using train1 as input to the Bayesian Regression equation (Equations 6). The similarity metric (Equation 9) was used in place of the Euclidean distance in Bayesian Regression (Equation 6).
2. Compute the linear regression parameters (w0, w1, w2, w3) by finding the best linear fit (Equation 8). Here you will need to use the ols function of statsmodels.formula.api. Your model should be fit using Δp1, Δp2, and Δp3 as the covariates. Note: the bitcoin order book data was not considered and hence rw4 term is not there in the model.
3. Use the linear regression model computed in Step 2 and Bayesian Regression estimates, to predict the price variations for the test dataset. Bayesian Regression estimates for test dataset are computed in the same way as they are computed for train2 dataset – using train1 as an input.
4. Once the price variations are predicted, compute the mean squared error (MSE) for the test dataset (the test dataset has 50 vectors => 50 predictions).

Reference paper: http://arxiv.org/pdf/1410.1231.pdf
