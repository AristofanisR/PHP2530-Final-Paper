data {
  int<lower=1> N;             // Number of observations
  int<lower=1> P;             // Number of features (pixels)
  int<lower=1> K;             // Number of latent components
  matrix[N, P] X;             // Centered data matrix
}

parameters {
  matrix[N, K] Z;                     // Latent scores
  matrix[P, K] W;                     // Loadings
  matrix<lower=0,upper=1>[P, K] theta; // Inclusion probabilities
  real<lower=0> slab_sd;              // Slab standard deviation
  real<lower=0> spike_sd;             // Spike standard deviation
  real<lower=0> sigma;                // Residual noise SD
}

model {
  // Priors
  to_vector(Z) ~ normal(0, 1);
  slab_sd ~ normal(0, 1);
  spike_sd ~ normal(0, 0.1);
  sigma ~ normal(0, 1);
  to_vector(theta) ~ beta(1, 1);  // Uniform prior on inclusion

  // Spike-and-slab prior on loadings
  for (j in 1:P) {
    for (k in 1:K) {
      real sd_jk = sqrt(square(slab_sd) * theta[j, k] + square(spike_sd) * (1 - theta[j, k]));
      W[j, k] ~ normal(0, sd_jk);
    }
  }

  // Likelihood
  for (n in 1:N)
    X[n] ~ normal(W * Z[n]', sigma);
}
