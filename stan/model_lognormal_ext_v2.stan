data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> R;
  array[N] int<lower=1, upper=S> station_id;
  array[S] int<lower=1, upper=R> river_id;
  vector<lower=0>[N] y;
  vector[S] log_catchment;   // ← NOWE: standaryzowany log(catchment_area)
}

parameters {
  real mu_global;
  real log_sigma_global;
  real alpha_catchment;      // ← NOWE: efekt catchment area na sigma

  real<lower=0> tau_mu_riv;
  real<lower=0> tau_sigma_riv;
  vector[R] z_mu_riv;
  vector[R] z_sigma_riv;

  real<lower=0> tau_mu_sta;
  real<lower=0> tau_sigma_sta;
  vector[S] mu_sta;
  vector[S] log_sigma_sta_raw;  // odchylenie od przewidywanej sigmy
}

transformed parameters {
  vector[R] mu_riv        = mu_global        + tau_mu_riv    * z_mu_riv;
  vector[R] log_sigma_riv = log_sigma_global + tau_sigma_riv * z_sigma_riv;

  // sigma stacji = efekt rzeki + efekt catchment area + losowy szum stacji
  vector[S] log_sigma_sta = log_sigma_riv[river_id]
                            + alpha_catchment * log_catchment
                            + tau_sigma_sta * log_sigma_sta_raw;

  vector[S] sigma_sta = exp(log_sigma_sta);
}

model {
  // Priory globalne
  mu_global        ~ normal(3, 1.2);
  log_sigma_global ~ normal(0, 0.5);
  alpha_catchment  ~ normal(0, 0.5);  // prior na efekt catchment

  // Skale hierarchii
  tau_mu_riv    ~ normal(0, 1.2);
  tau_sigma_riv ~ normal(0, 0.5);
  tau_mu_sta    ~ normal(0, 1.2);
  tau_sigma_sta ~ normal(0, 0.5);

  // Poziom rzeki (NCP)
  z_mu_riv    ~ normal(0, 1);
  z_sigma_riv ~ normal(0, 1);

  // Poziom stacji (CP dla mu, NCP dla sigma)
  mu_sta           ~ normal(mu_riv[river_id], tau_mu_sta);
  log_sigma_sta_raw ~ normal(0, 1);

  // Likelihood
  y ~ lognormal(mu_sta[station_id], sigma_sta[station_id]);
}