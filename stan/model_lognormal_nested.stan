data {
  int<lower=1> N;                                 // Całkowita liczba obserwacji
  int<lower=1> S;                                 // Liczba unikalnych stacji
  int<lower=1> R;                                 // Liczba unikalnych rzek
  array[N] int<lower=1, upper=S> station_id;      // Mapowanie: obserwacja -> stacja
  array[S] int<lower=1, upper=R> station_to_river; // Mapowanie: stacja -> rzeka
  vector<lower=0>[N] y;                           // Zaobserwowany przepływ Q [m3/s]
}

transformed data {
  // Logarytmowanie danych wejściowych raz przed startem samplowania
  vector[N] log_y = log(y);
}

parameters {
  // 1. Poziom globalny
  real mu_global;
  real log_sigma_global;
  
  // Zmienność na poziomie rzek (tau)
  real<lower=0> tau_mu_river;
  real<lower=0> tau_sigma_river;
  
  // Zmienność na poziomie stacji (tau)
  real<lower=0> tau_mu_station;
  real<lower=0> tau_sigma_station;

  // Zmienne pomocnicze do parametryzacji niescentrowanej (raw)
  vector[R] mu_river_raw;
  vector[R] log_sigma_river_raw;
  vector[S] mu_station_raw;
  vector[S] log_sigma_station_raw;
}

transformed parameters {
  vector[R] mu_river;
  vector[R] log_sigma_river;
  vector[S] mu_station;
  vector<lower=0>[S] sigma_station;

  // Konstrukcja poziomu rzek (niescentrowana)
  mu_river = mu_global + tau_mu_river * mu_river_raw;
  log_sigma_river = log_sigma_global + tau_sigma_river * log_sigma_river_raw;

  // Konstrukcja poziomu stacji (zagnieżdżenie w parametrach rzek)
  for (s in 1:S) {
    mu_station[s] = mu_river[station_to_river[s]] + tau_mu_station * mu_station_raw[s];
    sigma_station[s] = exp(log_sigma_river[station_to_river[s]] + tau_sigma_station * log_sigma_station_raw[s]);
  }
}

model {
  // Zbalansowane priory informatywne skalibrowane w PPC
  mu_global ~ normal(3, 1.2);
  log_sigma_global ~ normal(0, 0.5);
  
  tau_mu_river ~ normal(0, 1.2);
  tau_sigma_river ~ normal(0, 0.5);
  tau_mu_station ~ normal(0, 1.2);
  tau_sigma_station ~ normal(0, 0.5);

  // Rozkłady standardowe normalne dla parametrów pomocniczych
  mu_river_raw ~ std_normal();
  log_sigma_river_raw ~ std_normal();
  mu_station_raw ~ std_normal();
  log_sigma_station_raw ~ std_normal();

  // Funkcja wiarygodności (Likelihood)
  log_y ~ normal(mu_station[station_id], sigma_station[station_id]);
}

generated quantities {
  // Generowanie próbek z rozkładu posterior do testu PPC i porównania modeli
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = lognormal_rng(mu_station[station_id[n]], sigma_station[station_id[n]]);
  }
}