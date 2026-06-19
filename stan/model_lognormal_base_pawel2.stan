data {
  int<lower=1> N;                               // Liczba obserwacji dobowych
  int<lower=1> S;                               // Liczba stacji (69)
  array[N] int<lower=1, upper=S> station_id;    // Indeksy stacji
  vector<lower=0>[N] y;                         // Przepływy dobowe
}

parameters {
  real mu_global;                               // Globalna lokalizacja średniej kraju
  real<lower=0> tau_mu;                         // Zmienność średnich między stacjami
  real log_sigma_global;                        // Globalna baza log-wariancji szumu
  real<lower=0> tau_sigma;                      // Zmienność dynamiki między stacjami
  
  // W parametryzacji centrowanej parametry stacji deklarujemy bezpośrednio tutaj
  vector[S] mu;                                 
  vector[S] log_sigma;                          // Logarytm szumu, by po exp() był dodatni
}

transformed parameters {
  // Przekształcenie log-szumu do właściwej skali dla lognormal_lpdf
  vector[S] sigma = exp(log_sigma);
}

model {
  // 1. Priory globalne wyższego rzędu
  mu_global ~ normal(3, 1.2);
  tau_mu ~ normal(0, 1.2);
  log_sigma_global ~ normal(0, 0.5);
  tau_sigma ~ normal(0, 0.5);
  
  // 2. POZIOM HIERARCHII (PARAMETRYZACJA CENTROWANA)
  // Parametry stacji zależą bezpośrednio od parametrów globalnych jako ich średnich
  mu ~ normal(mu_global, tau_mu);
  log_sigma ~ normal(log_sigma_global, tau_sigma);

  // 3. ZWEKTORYZOWANY LIKELIHOOD
  y ~ lognormal(mu[station_id], sigma[station_id]);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = lognormal_rng(mu[station_id[n]], sigma[station_id[n]]);
    log_lik[n] = lognormal_lpdf(y[n] | mu[station_id[n]], sigma[station_id[n]]);
  }
}