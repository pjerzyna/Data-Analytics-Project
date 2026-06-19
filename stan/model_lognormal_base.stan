data {
  int<lower=1> N;                             // Calkowita liczba obserwacji (pomiarow)
  int<lower=1> S;                             // Liczba unikalnych stacji pomiarowych
  array[N] int<lower=1, upper=S> station_id;  // Indeks stacji dla kazdej obserwacji
  vector<lower=0>[N] y;                       // Zaobserwowany przeplyw Q [m3/s]
}

transformed data {
  // Logarytmowanie danych wejściowych raz przed startem samplowania
  vector[N] log_y = log(y);
}

parameters {
  // Parametry poziomu globalnego (ogólnokrajowego)
  real mu_global;
  real<lower=0> tau_mu;
  
  real log_sigma_global;
  real<lower=0> tau_sigma;
  
  // Zmienne pomocnicze do parametryzacji niescentrowanej
  vector[S] mu_raw;
  vector[S] sigma_raw; 
}

transformed parameters {
  // Parametry rzeczywiste dla kazdej stacji pomiarowej
  vector[S] mu;
  vector<lower=0>[S] sigma;
  
  // Konstrukcja parametru polozenia stacji (niescentrowana)
  mu = mu_global + tau_mu * mu_raw;
  
  // Konstrukcja parametru skali stacji przy uzyciu funkcji exp()
  sigma = exp(log_sigma_global + tau_sigma * sigma_raw);
}

model {
  // Zaktualizowane priory skalibrowane w tescie PPC
  mu_global ~ normal(3, 1.2); 
  tau_mu ~ normal(0, 1.2);
  log_sigma_global ~ normal(0, 0.5);
  tau_sigma ~ normal(0, 0.5);
  
  // Rozklady standardowe normalne dla zmiennych raw
  mu_raw ~ std_normal();
  sigma_raw ~ std_normal();
  
  // Funkcja wiarygodnosci (Likelihood) w przestrzeni zlogarytmizowanej
  log_y ~ normal(mu[station_id], sigma[station_id]);
}

generated quantities {
  // Generowanie probek z rozkladu posterior do testu Posterior Predictive Check
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = lognormal_rng(mu[station_id[n]], sigma[station_id[n]]);
  }
}