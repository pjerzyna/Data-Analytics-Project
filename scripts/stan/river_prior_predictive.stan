data {
  int<lower=1> N;   // liczba obserwacji
  int<lower=1> S;   // liczba stacji
  array[N] int<lower=1, upper=S> station; 
  array[N] int<lower=1, upper=12> month;  
  // nie przekazujemy tutaj wektora 'y' do bloku danych modelu, 
  // bo model priorów go nie potrzebuje do wnioskowania!
}

parameters {
  vector[12] mu_alpha;                // dla kazdego miesiaca inna logarytmiczna srednia przeplywu
  vector<lower=0>[12] sigma_alpha;    // dla kazdego miesiaca inna zmiennosc miedzy stacjami
  matrix[S, 12] alpha;                // dla kazdej stacji i miesiaca indywidualna srednia logarytmiczna przeplywu
  vector<lower=0>[S] sigma;           // dla kazdej stacji indywidualny szum wewnatrz stacji
}

model {
  // 1. Definiujemy nasze przemyślane priory (Założenia fizyczne)
  mu_alpha ~ normal(2.7, 1.5);          // Średnia wyciągnięta z analizy median
  sigma_alpha ~ exponential(2);         // Zmienność reżimów między stacjami
  
  for (s in 1:S) {
    for (m in 1:12) {
      alpha[s, m] ~ normal(mu_alpha[m], sigma_alpha[m]);
    }
  }
  sigma ~ exponential(2);           // Szum wewnątrz stacji
  
}

generated quantities {
  // Generujemy predykcje a priori (Sztuczne dane)
  vector[N] y_prior;
  for (n in 1:N) {
    real mu_log = alpha[station[n], month[n]];
    y_prior[n] = lognormal_rng(mu_log, sigma[station[n]]);
  }
}