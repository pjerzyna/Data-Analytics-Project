// Bez generated quantities, bo nie ma na to pamięci RAM - WYMAGA NAPRAWIENIA

data {
  int<lower=1> N;                              // Liczba obserwacji dobowych
  int<lower=1> S;                              // Liczba stacji (69)
  int<lower=1> R;                              // Liczba rzek
  array[N] int<lower=1, upper=S> station_id;   // Obserwacja --> stacja
  array[S] int<lower=1, upper=R> river_id;     // Stacja --> rzeka
  vector<lower=0>[N] y;                        // Przepływy dobowe
}

parameters {
  // --- POZIOM GLOBALNY ---
  real mu_global;
  real log_sigma_global;

  // --- POZIOM RZEKI (NCP) – zmienne pomocnicze w przestrzeni standaryzowanej ---
  real<lower=0> tau_mu_riv;
  real<lower=0> tau_sigma_riv;
  vector[R] z_mu_riv;
  vector[R] z_sigma_riv;

  // --- POZIOM STACJI (CP) – bezpośrednie losowanie z profilu rzeki ---
  real<lower=0> tau_mu_sta;
  real<lower=0> tau_sigma_sta;
  vector[S] mu_sta;
  vector[S] log_sigma_sta;
  // vector<lower=-4, upper=3>[S] log_sigma_sta;  // Dodanie granic, aby uniknąć ekstremalnych wartości sigm
}

transformed parameters {
  // Rekonstrukcja parametrów rzeki z NCP
  vector[R] mu_riv         = mu_global       + tau_mu_riv    * z_mu_riv;
  vector[R] log_sigma_riv  = log_sigma_global + tau_sigma_riv * z_sigma_riv;

  // Transformacja log-szumu do właściwej skali
  vector[S] sigma_sta = exp(log_sigma_sta);
}

model {
  // 1. PRIORY GLOBALNE
  mu_global        ~ normal(3, 1.2);
  log_sigma_global ~ normal(0, 0.5);

  // 2. SKALE HIERARCHII
  tau_mu_riv    ~ normal(0, 1.2);
  tau_sigma_riv ~ normal(0, 0.5);
  tau_mu_sta    ~ normal(0, 1.2);
  tau_sigma_sta ~ normal(0, 0.5);

  // 3. POZIOM RZEKI (NCP)
  // z są wolnymi zmiennymi – przesunięcie i skalowanie w transformed parameters
  z_mu_riv    ~ normal(0, 1);
  z_sigma_riv ~ normal(0, 1);

  // 4. POZIOM STACJI (CP)
  // Każda stacja losowana bezpośrednio z rozkładu swojej rzeki
  mu_sta       ~ normal(mu_riv[river_id],        tau_mu_sta);
  log_sigma_sta ~ normal(log_sigma_riv[river_id], tau_sigma_sta);

  // 5. LIKELIHOOD
  y ~ lognormal(mu_sta[station_id], sigma_sta[station_id]);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n]   = lognormal_rng(mu_sta[station_id[n]], sigma_sta[station_id[n]]);
    log_lik[n] = lognormal_lpdf(y[n] | mu_sta[station_id[n]], sigma_sta[station_id[n]]);
  }
}