data {
  int<lower=1> N;                                 // Calkowita liczba obserwacji
  int<lower=1> S;                                 // Liczba stacji
  int<lower=1> R;                                 // Liczba rzek
  array[N] int<lower=1, upper=S> station_id;      // Mapowanie: obserwacja -> stacja
  array[S] int<lower=1, upper=R> station_to_river; // Mapowanie: stacja -> rzeka
}

generated quantities {
  // 1. POZIOM GLOBALNY (Hiperparametry zgodne z kalibracja PPC modelu bazowego)
  real mu_global = normal_rng(3, 1.2);
  real log_sigma_global = normal_rng(0, 0.5);

  // Hiperparametry zmiennosci (tau) dla poszczegolnych poziomow
  real<lower=0> tau_mu_river = abs(normal_rng(0, 1.2));
  real<lower=0> tau_sigma_river = abs(normal_rng(0, 0.5));
  
  real<lower=0> tau_mu_station = abs(normal_rng(0, 1.2));
  real<lower=0> tau_sigma_station = abs(normal_rng(0, 0.5));

  // 2. POZIOM RZEKI (Generowanie parametrow dla kazdej rzeki r)
  vector[R] mu_river;
  vector[R] log_sigma_river;

  for (r in 1:R) {
    mu_river[r] = normal_rng(mu_global, tau_mu_river);
    log_sigma_river[r] = normal_rng(log_sigma_global, tau_sigma_river);
  }

  // 3. POZIOM STACJI (Generowanie parametrow dla kazdej stacji s zagniezdzonej w rzece)
  vector[S] mu_station;
  vector[S] sigma_station;

  for (s in 1:S) {
    mu_station[s] = normal_rng(mu_river[station_to_river[s]], tau_mu_station);
    sigma_station[s] = exp(normal_rng(log_sigma_river[station_to_river[s]], tau_sigma_station));
  }

  // 4. GENEROWANIE SYMULOWANYCH PRZEPŁYWÓW
  vector[N] y_sim;
  for (n in 1:N) {
    y_sim[n] = lognormal_rng(mu_station[station_id[n]], sigma_station[station_id[n]]);
  }
}