functions {
  // Własny generator liczb losowych dla rozkładu GEV (metoda odwróconej dystrybuanty)
  real gev_rng(real mu, real sigma, real xi) {
    real u = uniform_rng(0, 1);
    if (xi == 0) {
      return mu - sigma * log(-log(u));
    } else {
      return mu + (sigma / xi) * (pow(-log(u), -xi) - 1);
    }
  }
}

data {
  int<lower=1> S; // Liczba stacji (69)
  int<lower=1> T; // Liczba lat obserwacji maksimów (3)
}

parameters {
  real bar_mu;              // Globalna średnia lokalizacji maksimów
  real<lower=0> tau_mu;     // Zmienność parametru lokalizacji między stacjami
  
  real bar_sigma;           // Globalna średnia skali (w przestrzeni log)
  real<lower=0> tau_sigma;  // Zmienność parametru skali między stacjami
  
  vector[S] mu;             // Parametr lokalizacji dla każdej stacji (Centered)
  vector<lower=0>[S] sigma; // Parametr skali dla każdej stacji (Centered)
  
  real xi;                  // Współdzielony, globalny kształt ogona (shape)
}

model {
  // 1. Hiper-rozkłady a priori (Hyper-priors)
  bar_mu ~ normal(100, 50);       // Zakładamy, że średnie maksima oscylują wokół 100 m3/s
  tau_mu ~ exponential(0.02);     // Dość elastyczna zmienność lokalizacji między rzekami
  
  bar_sigma ~ normal(3, 1);       // exp(3) ~ 20 m3/s jako wyjściowy punkt zmienności fal
  tau_sigma ~ exponential(1);
  
  // 2. Rozkłady a priori dla parametrów stacji (Struktura Centered)
  mu ~ normal(bar_mu, tau_mu);
  sigma ~ lognormal(bar_sigma, tau_sigma);
  
  // 3. Prior dla kształtu ogona (Słabo informacyjny, dopuszcza ciężkie ogony Frécheta)
  xi ~ normal(0.1, 0.2); 
}

generated quantities {
  // Generowanie danych syntetycznych (Prior Predictive Simulation)
  matrix[S, T] y_sim;
  
  for (s in 1:S) {
    for (t in 1:T) {
      y_sim[s, t] = gev_rng(mu[s], sigma[s], xi);
    }
  }
}