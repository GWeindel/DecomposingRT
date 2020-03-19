# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import pystan
import stan_utility

##### Recovering data from notebook 0
data = pd.read_csv('../trimmed_data.csv')
LMEdata = data.copy() 

##### log transform
LMEdata["logrt"] = np.log(LMEdata.rt)
LMEdata["logpmt"] = np.log(LMEdata.pmt)
LMEdata["logmt"] = np.log(LMEdata.mt)

##### Contrast coding
LMEdata['contrast'] = LMEdata.contraste.replace([.01,.025,.07,.15,.30], [-.5,-.25,0,.25,.5])
LMEdata['SAT'] = LMEdata.apply(lambda row: 0.5 if row['condition'] == 'Accuracy' else -0.5, axis=1)
LMEdata['givenResp'] = LMEdata.apply(lambda row: -0.5 if row['givenResp'] == 'Right' else 0.5, axis=1)

##### Stan specific adjustments
LMEdata['response'] = LMEdata.apply(lambda row: 0 if row['response'] == 1 else 1, axis=1)
LMEdata['participant'] = [int(x.split('S')[-1].split('_')[0]) for x in LMEdata.participant]

LMEdata = LMEdata[["logrt","logpmt","logmt","participant", "response",
                   "SAT","contrast","exp", "givenResp"]]
LMEdata.to_csv('LMEdata.csv')

##### Writing Stan program for LME
LME = """
data {
  int<lower=0> N;               //N observations total
  int<lower=1> P;               //N population-level effects  (intercept + beta)
  int<lower=0> J;               //N subj
  int<lower=1> n_u;             //N subj ranefs (ex. x * x | part = 4) if random effect on all effects, same argument as P can be passed
  int<lower=1,upper=J> subj[N]; //subj ID vector
  matrix[N,P] X;                //fixef design matrix
  matrix[N, n_u] Z_u;           //subj ranef design matrix, X if raneff = fixeff
  vector[N] y;                  // DV
  vector[2] p_intercept;        // prior on intercept
  vector[2] p_sd;               // prior on residual std
  vector[P-1] p_fmu;            // priors on fixef mu
  vector[P-1] p_fsigma;         // priors on fixef sigma
  vector[2] p_r;                // priors on ranef
  int<lower=0, upper=1> logT;                     // are the data log transformed ?
}

transformed data {
  matrix[N,P-1] X_beta; // removing intercept
  X_beta = block(X,1,2,N,P-1); // returns X from row 1 and col 2 to N rows and P-1 columns
}

parameters {
  real alpha;                   // intercept
  real<lower=0> sigma;          //residual std
  vector[P-1] beta;             //population-level effects coefs (w\o intercept)
  cholesky_factor_corr[n_u] L_u;//cholesky factor of subj ranef corr matrix
  vector<lower=0>[n_u] sigma_u; //subj ranef std INCLUDING INTERCEPT, hence beta[1] ~ sigma[2]
  vector[n_u] z_u[J];           //spherical subj ranef
}

transformed parameters {
  vector[n_u] u[J];             //subj ranefs
  {
    matrix[n_u,n_u] Sigma_u;    //subj ranef cov matrix
    Sigma_u = diag_pre_multiply(sigma_u,L_u);
    for(j in 1:J)
      u[j] = Sigma_u * z_u[j];
  }
}

model {
  vector[N] mu;
  //priors
  L_u ~ lkj_corr_cholesky(2.0);
  alpha ~ normal(p_intercept[1], p_intercept[2]);
  beta ~ normal(p_fmu, p_fsigma);
  sigma ~ normal(p_sd[1], p_sd[2]);
  for (j in 1:J)
    z_u[j] ~ normal(p_r[1],p_r[2]);

  //likelihood
  for (n in 1:N)
    mu[n] = alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]];
  y ~ normal(mu, sigma);
}

generated quantities {
  matrix[n_u,n_u] Cor_u;
  vector[N] log_lik;
  vector[N] y_hat;              // predicted y
  real raw_intercept;           //raw intercept if log values
  vector[P-1] raw_beta;         //raw effect size if log values
  Cor_u = tcrossprod(L_u);      //Correlations between random effects by subj
  if (logT == 1) {
    raw_intercept = exp(alpha);
    raw_beta = exp(alpha + beta) - raw_intercept;
  }
  for (n in 1:N){
    log_lik[n] = normal_lpdf(y[n] | alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]], sigma);
    y_hat[n] = normal_rng(alpha + X_beta[n] * beta + Z_u[n] * u[subj[n]], sigma);
  }
}
"""
f= open("LME.stan","w+")
f.write(LME)
f.close()
LME = stan_utility.compile_model('LME.stan', model_name="LME")
