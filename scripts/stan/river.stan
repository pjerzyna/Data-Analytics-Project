data {
  int<lower=1> N;                // Całkowita liczba obserwacji (np. 365 dni * 69 stacji)
  int<lower=1> S;                // Liczba stacji (69)
  array[N] int<lower=1, upper=S> station; // Indeks stacji dla każdej obserwacji (od 1 do 69)
  array[N] int<lower=1, upper=12> month;  // Indeks miesiąca dla każdej obserwacji (od 1 do 12)
  vector<lower=0>[N] y;          // Przepływ w danym dniu (kolumna Value z roku 2023)
}

parameters {
  // Globalne hiperparametry dorzecza - osobne dla każdego z 12 miesięcy
  vector[12] mu_alpha;
  vector<lower=0>[12] sigma_alpha;
  
  // Parametry indywidualne dla stacji s w miesiącu m (Partial Pooling)
  matrix[S, 12] alpha;           
  vector<lower=0>[S] sigma;      // Ogólna zmienność (szum) dla każdej stacji
}

model {
  // 1. Hiperpriory (charakterystyka ogólna dorzecza dla poszczególnych miesięcy)
  mu_alpha ~ normal(0, 5);
  sigma_alpha ~ exponential(1);
  
  // 2. Priory hierarchiczne (Tutaj zachodzi Partial Pooling)
  for (s in 1:S) {
    for (m in 1:12) {
      alpha[s, m] ~ normal(mu_alpha[m], sigma_alpha[m]);
    }
  }
  sigma ~ exponential(1);
  
  // 3. Funkcja wiarygodności (Likelihood - modelowanie profilu miesięcznego)
  for (n in 1:N) {
    real mu_log = alpha[station[n], month[n]];
    y[n] ~ lognormal(mu_log, sigma[station[n]]);
  }
}

generated quantities {
  // Generowanie predykcji (Posterior Predictive Checks) do weryfikacji dopasowania do roku 2023
  vector[N] y_rep;
  for (n in 1:N) {
    real mu_log = alpha[station[n], month[n]];
    y_rep[n] = lognormal_rng(mu_log, sigma[station[n]]);
  }
}