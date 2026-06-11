data {
  int<lower=1> N;                // Całkowita liczba obserwacji maksimów (np. ~200)
  int<lower=1> S;                // Liczba stacji (69)
  vector[N] y;                   // Wektor wszystkich maksimów rocznych
  array[N] int<lower=1, upper=S> station_id; // Mapowanie: wiersz -> ID stacji
}

parameters {
  real bar_mu;                  // Średnia globalna parametru lokalizacji
  real<lower=0> tau_mu;         // Zmienność lokalizacji między stacjami
  
  real bar_sigma;               // Średnia globalna skali (w przestrzeni log)
  real<lower=0> tau_sigma;      // Zmienność skali między stacjami
  
  vector[S] mu_raw;             // Surowy szum lokalizacji (NON-CENTERED)
  vector[S] sigma_raw;          // Surowy szum skali (NON-CENTERED)
  
  real xi;                      // Globalny kształt ogona (współdzielony)
}

transformed parameters {
  vector[S] mu;
  vector<lower=0>[S] sigma;
  
  // Matematyczne rozprostowanie geometrii lejeka Neala
  mu = bar_mu + tau_mu * mu_raw;
  for (s in 1:S) {
    sigma[s] = exp(bar_sigma + tau_sigma * sigma_raw[s]);
  }
}

model {
  // 1. Priory hiperparametrów
  bar_mu ~ normal(100, 50);
  tau_mu ~ exponential(0.02);
  bar_sigma ~ normal(3, 1);
  tau_sigma ~ exponential(1);
  
  // 2. Priory dla surowych wektorów (Standard normal generuje non-centered)
  mu_raw ~ normal(0, 1);
  sigma_raw ~ normal(0, 1);
  
  // Prior kształtu ogona
  xi ~ normal(0.1, 0.2);
  
  // 3. Likelihood: Uogólniony Rozkład Wartości Ekstremalnych (GEV)
  for (n in 1:N) {
    real m_n = mu[station_id[n]];
    real s_n = sigma[station_id[n]];
    
    if (xi == 0) {
      real z = (y[n] - m_n) / s_n;
      target += -log(s_n) - z - exp(-z);
    } else {
      real t = 1.0 + xi * (y[n] - m_n) / s_n;
      if (t > 0) {
        target += -log(s_n) - (1.0 + 1.0 / xi) * log(t) - pow(t, -1.0 / xi);
      } else {
        target += -negative_infinity(); // Odrzucenie matematycznie niemożliwych kombinacji
      }
    }
  }
}

generated quantities {
  // Replikacja danych do późniejszego Posterior Predictive Check
  vector[N] y_rep;
  
  for (n in 1:N) {
    real m_n = mu[station_id[n]];
    real s_n = sigma[station_id[n]];
    real u = uniform_rng(0, 1);
    
    if (xi == 0) {
      y_rep[n] = m_n - s_n * log(-log(u));
    } else {
      y_rep[n] = m_n + (s_n / xi) * (pow(-log(u), -xi) - 1);
    }
  }
}