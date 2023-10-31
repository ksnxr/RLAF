data {
  int<lower=0> N;
  matrix[N, 1] x;
  vector[N] y;
  real sigma_noise;
  real prior_std;
}
parameters {
  matrix[1, 10] W1;
  row_vector[10] b1;
  matrix[10, 1] W2;
  real b2;
}
model {
  to_vector(W1) ~ normal(0.0, prior_std);
  b1 ~ normal(0.0, prior_std);
  to_vector(W2) ~ normal(0.0, prior_std);
  b2 ~ normal(0.0, prior_std);
  y ~ normal(to_vector(tanh(x * W1 + rep_vector(1.0, N) * b1) * W2 + b2), sigma_noise);
}