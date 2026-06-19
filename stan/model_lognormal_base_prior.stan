data {
  int<lower=1> N;             // Calkowita liczba obserwacji
  int<lower=1> S;             // Liczba stacji
  array[N] int<lower=1, upper=S> station_id; 
}

generated quantities {
  // ZAKTUALIZOWANE, ZBALANSOWANE PRIORY
  
  // mu_global: Średni przepływ polskich rzek w skali log. 
  // Rozluźniona wariancja z 1.0 na 1.2, daje więcej oddechu dla większych rzek.
  real mu_global = normal_rng(3, 1.2);
  
  // tau_mu: Zmienność między stacjami 
  real<lower=0> tau_mu = abs(normal_rng(0, 1.2));
  
  // log_sigma_global: Bazowa wariancja pomiarów. 
  // Średnia wraca do zera (neutralna), wąskie odchylenie zapobiega eksplozji.
  real log_sigma_global = normal_rng(0, 0.5);
  
  // tau_sigma: Zmienność wariancji między stacjami
  real<lower=0> tau_sigma = abs(normal_rng(0, 0.5));
  
  vector[S] mu_station;
  vector[S] sigma_station;
  
  for (s in 1:S) {
    mu_station[s] = normal_rng(mu_global, tau_mu);
    sigma_station[s] = exp(normal_rng(log_sigma_global, tau_sigma));
  }
  
  vector[N] y_sim;
  for (n in 1:N) {
    y_sim[n] = lognormal_rng(mu_station[station_id[n]], sigma_station[station_id[n]]);
  }
}