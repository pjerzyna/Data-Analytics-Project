data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1, upper=S> station;
  array[N] int<lower=1, upper=12> month;
}

parameters {
  vector[12] mu_alpha;
  vector<lower=0>[12] sigma_alpha;

  // Non-centered: zamiast alpha ~ normal(mu, sigma),
  // parametryzujemy przez surowe odchylenia z ~ normal(0,1)
  matrix[S, 12] z_alpha;

  // Sigma wewnątrz stacji — ograniczona do rozsądnego zakresu
  vector<lower=0>[S] sigma;
}

transformed parameters {
  // Dopiero tutaj rekonstruujemy alpha z komponentów
  matrix[S, 12] alpha;
  for (s in 1:S)
    for (m in 1:12)
      alpha[s, m] = mu_alpha[m] + sigma_alpha[m] * z_alpha[s, m];
}

model {
  // Hyperpriors
  mu_alpha    ~ normal(2.7, 1.0);   // log(15) ≈ 2.7 — środek skali polskich rzek
  sigma_alpha ~ exponential(4);     // mean=0.25 — zmienność między stacjami w log-skali

  // Non-centered: surowe odchylenia mają prior N(0,1)
  to_vector(z_alpha) ~ normal(0, 1);

  // Szum wewnątrz stacji — HalfNormal zamiast Exponential, cięższy ogon bardziej kontrolowany
  // mean ≈ 0.4, max realnie ~0.8-1.0
  sigma ~ normal(0, 0.5);
}

generated quantities {
  vector[N] y_prior;
  for (n in 1:N) {
    real mu_log = alpha[station[n], month[n]];
    y_prior[n] = lognormal_rng(mu_log, sigma[station[n]]);
  }
}