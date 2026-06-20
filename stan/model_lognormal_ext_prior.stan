data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> R;
  array[N] int<lower=1, upper=S> station_id;
  array[S] int<lower=1, upper=R> river_id;
  vector<lower=0>[N] y;
}

generated quantities {
  // Hiperparametry globalne
  real mu_global        = normal_rng(3, 1.2);
  real log_sigma_global = normal_rng(0, 0.5);
  real tau_mu_riv       = abs(normal_rng(0, 1.2));
  real tau_sigma_riv    = abs(normal_rng(0, 0.5));
  real tau_mu_sta       = abs(normal_rng(0, 1.2));
  real tau_sigma_sta    = abs(normal_rng(0, 0.5));

  // Poziom rzeki (NCP)
  vector[R] z_mu_riv;
  vector[R] z_sigma_riv;
  vector[R] mu_riv;
  vector[R] log_sigma_riv;
  for (r in 1:R) {
    z_mu_riv[r]    = normal_rng(0, 1);
    z_sigma_riv[r] = normal_rng(0, 1);
    mu_riv[r]      = mu_global + tau_mu_riv * z_mu_riv[r];
    log_sigma_riv[r] = log_sigma_global + tau_sigma_riv * z_sigma_riv[r];
  }

  // Poziom stacji (CP)
  vector[S] mu_sta;
  vector[S] log_sigma_sta;
  for (s in 1:S) {
    mu_sta[s]        = normal_rng(mu_riv[river_id[s]], tau_mu_sta);
    log_sigma_sta[s] = normal_rng(log_sigma_riv[river_id[s]], tau_sigma_sta);
  }

  // Symulacja przepływów
  vector[N] y_sim;
  for (n in 1:N) {
    y_sim[n] = lognormal_rng(
      mu_sta[station_id[n]],
      exp(log_sigma_sta[station_id[n]])
    );
  }
}