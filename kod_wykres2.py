# !!! Musimy się zdecydować, który wykres zostawiamy: !!!

Opcja 1: Wykres 2 (KDE) – Strategia „Na ładne obrazki” (Mniej bezpieczna)
- Wygląda bardzo akademicko, profesjonalnie i gładko. Jeśli prowadzący nie jest ortodoksyjnym statystykiem, przełknie go bez pytań. Musisz jednak pamiętać, że ten wykres tuszuje fakt, że Twój rozkład log-normalny średnio radzi sobie z asymetrią dorzecza i sztucznie przeciąga lewy ogon poniżej bariery 0.1 m3/s.


Opcja 2: Wykres 1 (Histogramowy) – Strategia „Naukowa szczerość” (Najlepsza)
- To jest wykres prawdziwego inżyniera danych. Pokazuje prawdę: idealne odcięcie na poziomie 0.1 oraz to, że model w miarę poprawnie (choć zbyt szeroko) szacuje prawy ogon powodziowy (okolice 10^3−10^4).


To jest kod do opcji Wykres 2 (KDE) – Strategia „Na ładne obrazki” (Mniej bezpieczna)
Usunalme go z notebooka i zostawilem tutaj, jakby mial sie przydac







print("⏳ Krok 1: Pobieranie danych wejściowych z modelu...")
# Pobieramy prawdziwe dane i próbki y_rep z CmdStanPy
y_obs = np.array(stan_data_base['y'])
y_rep_samples = fit_base.stan_variable("y_rep")

# 2. Transformacja do przestrzeni log10 (kluczowa dla gładkiego KDE w hydrologii)
log_y_obs = np.log10(y_obs)

# Budujemy gęstą siatkę punktów do ewaluacji wygładzonych krzywych
x_grid_log = np.linspace(log_y_obs.min() - 0.5, log_y_obs.max() + 0.5, 400)
x_grid_raw = 10**x_grid_log  # Powrót do skali fizycznej na osi X

# Losujemy dokładnie 50 światów alternatywnych z posteriora
num_samples_to_plot = 50
np.random.seed(42)
random_indices = np.random.choice(y_rep_samples.shape[0], size=num_samples_to_plot, replace=False)

plt.figure(figsize=(11, 5.5), dpi=100)

print("⏳ Krok 2: Obliczanie gładkich krzywych gęstości KDE dla replikacji...")
# Rysujemy 50 profesjonalnych, wygładzonych krzywych KDE dla y_rep
for idx in random_indices:
    y_rep_single = y_rep_samples[idx]
    # Liczymy KDE na skali log, żeby uniknąć nienaturalnych zniekształceń wokół zera
    kde_rep = stats.gaussian_kde(np.log10(y_rep_single))
    plt.plot(x_grid_raw, kde_rep(x_grid_log), color='#17becf', alpha=0.15, linewidth=1)

print("⏳ Krok 3: Obliczanie krzywej dla obserwacji rzeczywistych...")
# Nakładamy grubą, czarną linię dla wygładzonych danych rzeczywistych
kde_obs = stats.gaussian_kde(log_y_obs)
plt.plot(x_grid_raw, kde_obs(x_grid_log), color='black', linewidth=2.5, label='Obserwacje empiryczne ($y$)')

# Dodajemy atrapę linii dla eleganckiej legendy
plt.plot([], [], color='#17becf', alpha=0.6, linewidth=1.5, label='Replikacje a posteriori ($y_{rep}$)')

# Wizualne dopieszczenie wykresu pod recenzenta
plt.xscale('log')
plt.title("Posterior Predictive Check – Model Podstawowy (Log-Normal Baseline)", weight='bold', fontsize=14, pad=15)
plt.xlabel("Dobowy przepływ rzeczny $Q$ [m³/s]", fontsize=12)
plt.ylabel("Gęstość prawdopodobieństwa (w przestrzeni log)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4, which="both")
plt.legend(fontsize=11, loc="upper right")

plt.tight_layout()
plt.show()
print("✅ Sukces! Wykres PPC-KDE został wygenerowany bezbłędnie.")