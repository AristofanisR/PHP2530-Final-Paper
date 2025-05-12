data {
  int<lower=1> N;             // Number of observations
  int<lower=1> P;             // Number of variables (pixels)
  int<lower=1> K;             // Number of latent components
  matrix[N, P] X;             // Centered data matrix
}

parameters {
  matrix[N, K] Z;                      // Latent scores
  matrix[P, K] W;                      // Loadings
  vector<lower=0>[K] tau;             // ARD shrinkage precisions
  real<lower=0> sigma;                // Noise std deviation
}

model {
  // Priors
  tau ~ gamma(1, 1);
  for (k in 1:K)
    W[, k] ~ normal(0, 1 / sqrt(tau[k]));

  to_vector(Z) ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // Likelihood: each row of X ~ N(W * Z', sigma)
  for (n in 1:N) {
    row_vector[P] mu = (Z[n] * W');  // row_vector × matrix
    X[n] ~ normal(mu, sigma);
  }
}
