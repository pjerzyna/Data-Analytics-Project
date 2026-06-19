data {
  int<lower=1> N;                               // Całkowita liczba obserwacji dobowych
  int<lower=1> S;                               // Liczba unikalnych stacji pomiarowych (69)
  array[N] int<lower=1, upper=S> station_id;    // Wektor indeksów stacji dla każdej obserwacji
  vector<lower=0>[N] y;                         // Rzeczywiste przepływy dobowe [m³/s]
}

parameters {
  real mu_global;                               // Globalna lokalizacja średniej kraju
  real<lower=0> tau_mu;                         // Zmienność średnich między stacjami
  real log_sigma_global;                        // Globalna baza log-wariancji szumu
  real<lower=0> tau_sigma;                      // Zmienność dynamiki (szumu) między stacjami
  
  vector[S] mu_raw;                             // Standaryzowane błędy dla lokalizacji stacji
  vector[S] sigma_raw;                          // Standaryzowane błędy dla skali szumu stacji
}

transformed parameters {
  vector[S] mu;
  vector[S] sigma;
  
  // Pełna parametryzacja niescentrowana (Non-centered parameterization)
  // Przeciwdziała powstawaniu lejka Neala i zapewnia stabilność numeryczną
  for (s in 1:S) {
    mu[s] = mu_global + tau_mu * mu_raw[s];
    sigma[s] = exp(log_sigma_global + tau_sigma * sigma_raw[s]);
  }
}

model {
  // --- Zaktualizowane, wykalibrowane rozkłady a priori (zgodnie z PPC) ---
  mu_global ~ normal(3, 1.2);
  tau_mu ~ normal(0, 1.2);                       // Ponieważ parametrowi nadano lower=0, jest to Half-Normal
  log_sigma_global ~ normal(0, 0.5);
  tau_sigma ~ normal(0, 0.5);                    // Half-Normal dzięki ograniczeniu lower=0
  
  // Priory dla zmiennych standaryzowanych przestrzeni matematycznej
  mu_raw ~ normal(0, 1);
  sigma_raw ~ normal(0, 1);

  // --- ZWEKTORYZOWANA FUNKCJA WIARYGODNOŚCI (LIKELIHOOD) ---
  // Stan rozszerzy wektory mu i sigma za pomocą indeksów station_id w ułamku sekundy
  y ~ lognormal(mu[station_id], sigma[station_id]);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  
  // Generowanie próbek do Posterior Predictive Check (PPC) oraz Leave-One-Out CV
  for (n in 1:N) {
    y_rep[n] = lognormal_rng(mu[station_id[n]], sigma[station_id[n]]);
    log_lik[n] = lognormal_lpdf(y[n] | mu[station_id[n]], sigma[station_id[n]]);
  }
}