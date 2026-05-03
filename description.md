# Opis projektu — Cyber Security Attacks Classifier

Dokument w przystępny sposób tłumaczy, **co dokładnie robi projekt**, **jak działa krok po kroku** oraz **jaki jest jego sens**. Tekst jest wyłącznie po polsku i ma stanowić uzupełnienie istniejącego pliku [README.md](README.md) oraz dokumentacji w katalogu [docs/](docs/).

---

## 1. Cel i sens projektu

Projekt rozwiązuje problem **klasyfikacji wieloklasowej (multiclass classification)** zdarzeń sieciowych. Na podstawie cech opisujących pojedynczy pakiet/zdarzenie sieciowe (porty, protokół, długość pakietu, sygnatura ataku, poziom ważności itd.) model uczenia maszynowego ma przypisać próbkę do **jednej z trzech klas ataku**:

- **DDoS** (rozproszona odmowa usługi),
- **Malware** (oprogramowanie złośliwe),
- **Intrusion** (włamanie sieciowe).

Sens praktyczny: w środowisku **SOC (Security Operations Center)** analitycy codziennie mają do czynienia z dużą liczbą logów i zdarzeń. Automatyczna klasyfikacja typu incydentu pozwoliłaby (przy odpowiedniej jakości modelu) **wstępnie segregować zdarzenia**, kierować je do właściwych zespołów reagujących i przyspieszać reakcję.

Sens edukacyjny (główny dla tego projektu uczelnianego): pokazać **kompletny, powtarzalny pipeline ML** od pobrania danych aż po dashboard z wynikami — z uczciwą, krytyczną interpretacją rezultatów.

---

## 2. Zbiór danych

- **Źródło:** [Cyber Security Attacks na Kaggle](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks).
- **Rozmiar:** **40 000 wierszy × 25 kolumn**.
- **Zmienna docelowa:** `Attack Type` z trzema niemal idealnie zbalansowanymi klasami (~33% każda).
- **Charakter cech:** numeryczne (porty, długość pakietu, wynik anomalii), kategoryczne (protokół, typ pakietu, segment sieci, poziom ważności…), tekstowe (payload, user-agent), pola z dużą ilością braków (~50% — np. wskaźniki malware, alerty), kolumny identyfikujące (znacznik czasu, adresy IP).
- Pobieranie odbywa się automatycznie przy pierwszym uruchomieniu — odpowiada za to skrypt [download_data.py](download_data.py), który korzysta z `kagglehub` i zapisuje plik CSV do katalogu [data/](data/).

---

## 3. Architektura projektu

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  download_data.py    │ → │      pipeline.py     │ → │        app.py        │
│  pobranie danych z   │   │  EDA, preprocessing, │   │  dashboard Streamlit │
│  Kaggle (kagglehub)  │   │  trening, ewaluacja  │   │  (przeglądanie       │
│                      │   │  (zapis do results/) │   │   wyników)           │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

| Plik | Rola |
|------|------|
| [download_data.py](download_data.py) | Pobranie zbioru z Kaggle, zapis CSV do `data/` |
| [pipeline.py](pipeline.py) | Cały potok ML: EDA, preprocessing, trening, ewaluacja, zapis wyników |
| [app.py](app.py) | Interaktywny dashboard Streamlit prezentujący dane i wyniki |
| [start.sh](start.sh) / [start.bat](start.bat) | Jednokomendowe uruchomienie wszystkiego |
| `results/` | Wygenerowane metryki (JSON) i wykresy (PNG) — używa ich dashboard |
| `models/` | Zapisany wytrenowany model Random Forest (`random_forest.joblib`) |

---

## 4. Co robi projekt — krok po kroku

### Krok 1. Pobranie danych

Skrypt [download_data.py](download_data.py) sprawdza, czy `data/cybersecurity_attacks.csv` już istnieje. Jeśli nie — pobiera dataset przez `kagglehub` i kopiuje plik CSV do katalogu `data/`. Dzięki temu cały pipeline startuje od razu, bez ręcznej konfiguracji.

### Krok 2. Eksploracyjna analiza danych (EDA)

Funkcja `run_eda()` w [pipeline.py](pipeline.py) wykonuje **6 podkroków** i zapisuje wykresy do `results/`:

1. **Rozkład klas** — sprawdzamy, czy dataset jest zbalansowany (jest: ~33% każda).
2. **Braki w danych** — liczba i % braków per kolumna; identyfikujemy kolumny z ~50% braków.
3. **Histogramy cech numerycznych** — porty, długość pakietu, wyniki anomalii.
4. **Wykresy słupkowe cech kategorycznych** — protokół, typ pakietu, severity itd.
5. **Macierz korelacji** cech numerycznych (Pearson).
6. **Mutual Information (MI) z `Attack Type`** — wyłapuje zarówno zależności liniowe, jak i nieliniowe; ważny krok diagnostyczny mówiący, które cechy w ogóle niosą sygnał o klasie.

Wszystkie wyniki EDA trafiają też do `eda_summary.json`.

### Krok 3. Preprocessing (przygotowanie danych do uczenia)

Funkcja `preprocess()` realizuje **6 etapów**:

1. **Usunięcie kolumn nieprzydatnych do generalizacji** (10 kolumn): znacznik czasu, adresy IP (źródło/cel), `Payload Data`, `User Information`, `Device Information`, `Geo-location Data`, `Proxy Information`, `Firewall Logs`, `IDS/IPS Alerts`. Powód: pola wolnotekstowe, identyfikatory albo logi z dużą ilością braków, które nie generalizują na nowe dane.
2. **Obsługa braków** dla kolumn z ~50% NaN (`Malware Indicators`, `Alerts/Warnings`) — zamiana na **flagi binarne** (0/1) sygnalizujące „obecne / nieobecne”.
3. **Kodowanie zmiennej docelowej** (`LabelEncoder`): `DDoS=0`, `Intrusion=1`, `Malware=2`.
4. **Kodowanie cech kategorycznych** (`LabelEncoder`) — wystarczające dla modeli drzewiastych.
5. **Standaryzacja cech numerycznych** (`StandardScaler`) — `Source Port`, `Destination Port`, `Packet Length`, `Anomaly Scores`. Skalowanie głównie pomaga modelom odległościowym (k-NN) i jest neutralne dla drzew.
6. **Podział train/test 80/20 ze stratyfikacją** względem klasy → **32 000 próbek treningowych** i **8 000 testowych**.

Po preprocessingu do modelu wchodzi **14 cech** (wszystkie zachowane kolumny poza usuniętymi i poza targetem). Pełna konfiguracja jest zapisywana w `preprocessing_info.json`.

### Krok 4. Trening modeli

Funkcja `train_models()` trenuje równolegle **trzy klasyfikatory** na tym samym podziale danych:

1. **Random Forest (model główny)** — `n_estimators=200`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`. Model jest zapisywany do `models/random_forest.joblib`.
2. **Gradient Boosting (porównawczy)** — `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`.
3. **k-NN (porównawczy)** — `k=7`.

Dla każdego modelu liczona jest **5-krotna walidacja krzyżowa (5-fold CV)** na zbiorze treningowym (stabilniejsza ocena niż jeden podział).

### Krok 5. Ewaluacja na zbiorze testowym (20%)

Funkcja `evaluate()` dla **Random Forest** liczy szczegółowe metryki i generuje wykresy:

- **Metryki:** accuracy, F1 (macro i weighted), precision (macro), recall (macro), ROC AUC w trybie *one-vs-rest* (macro).
- **Raport klasyfikacji per klasa** (precision/recall/F1/support).
- **Macierz pomyłek** w dwóch wariantach: liczbowa i znormalizowana (procentowa).
- **Krzywe ROC (One-vs-Rest)** — po jednej dla każdej klasy.
- **Ważność cech (feature importance)** z lasu losowego.

Wszystkie pozostałe modele są oceniane na zbiorze testowym (te same metryki) — wyniki trafiają do `model_comparison.json` i wspólnego wykresu porównawczego.

### Krok 6. Wizualizacja w dashboardzie Streamlit

Plik [app.py](app.py) udostępnia dashboard z **8 zakładkami**:

| Zakładka | Co pokazuje |
|----------|-------------|
| 📋 Project Overview | Skrót problemu, tabela atrybutów, opis modelu głównego |
| 📊 Exploratory Data Analysis | Statystyki, próbka danych, rozkład klas, braki, histogramy/boxploty, korelacje |
| ⚙️ Preprocessing | Wszystkie kroki przygotowania danych + lista cech końcowych |
| 🌲 Model & Training | Hiperparametry RF, wynik 5-fold CV per fold + średnia |
| ⚖️ Model Comparison | RF vs Gradient Boosting vs k-NN — accuracy, F1, precision, recall, ROC AUC |
| 📈 Results & Evaluation | Szczegóły dla RF: macierz pomyłek, ROC, ważność cech, raport per klasa |
| 🔍 Interactive Explorer | Filtrowanie surowych danych, scatter plot, rozkłady wg klasy |
| ℹ️ Informacje | Przewodnik po projekcie po polsku |

Dashboard **odczytuje gotowe artefakty z `results/`** — nie trenuje modelu na nowo. Dlatego najpierw należy uruchomić `pipeline.py`.

---

## 5. Uzyskane wyniki — krótka interpretacja

Po pełnym uruchomieniu pipeline'u (Random Forest, zbiór testowy 8 000 próbek):

| Metryka | Wartość |
|---------|---------|
| Accuracy | **0,332** |
| F1 (macro) | 0,332 |
| Precision (macro) | 0,332 |
| Recall (macro) | 0,332 |
| ROC AUC (OvR, macro) | **0,498** |
| CV mean (5-fold) | 0,336 ± 0,005 |

Porównanie modeli (test):

| Model | Accuracy | F1 (macro) | ROC AUC |
|-------|----------|-----------|---------|
| Random Forest | 0,332 | 0,332 | 0,498 |
| Gradient Boosting | **0,339** | 0,339 | 0,505 |
| k-NN (k=7) | 0,329 | 0,326 | 0,496 |

**Co to oznacza?** Wszystkie trzy modele osiągają wynik bardzo bliski **losowego zgadywania** (1/3 ≈ 0,333) i **ROC AUC ≈ 0,5**. Kluczowy wniosek: po usunięciu identyfikatorów i pól tekstowych **pozostałe cechy w tym konkretnym zbiorze nie niosą wystarczającego sygnału** do rozróżnienia klas. Potwierdza to wykres *Mutual Information* (wszystkie cechy mają MI bliskie zera) oraz feature importance lasu (cechy mają niemal identyczną wagę — model dzieli „po równo”, bo nie ma czego się uchwycić).

To jest **wynik metodologicznie poprawny i wartościowy**: pokazuje, że dataset (mimo etykiety „cyber security”) jest prawdopodobnie wygenerowany syntetycznie bez prawdziwej zależności cech od klasy. Kod, pipeline i dashboard działają poprawnie — to dane są ograniczeniem, nie metoda.

---

## 6. Jak uruchomić

### Uruchomienie jedną komendą

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```bat
start.bat
```

Skrypt sam:
1. wykryje Pythona (3.10–3.13),
2. utworzy `venv/` i zainstaluje `requirements.txt`,
3. uruchomi `pipeline.py` (pobranie danych + trening + zapis wyników),
4. wystartuje Streamlit pod adresem `http://localhost:8501`.

### Uruchomienie ręczne

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py                # generuje results/ i models/
streamlit run app.py              # uruchamia dashboard
```

---

## 7. Co znajduje się w `results/`

Po uruchomieniu pipeline'u:

| Plik | Zawartość |
|------|-----------|
| `metrics.json` | Wszystkie metryki RF + raport klasyfikacji |
| `model_comparison.json` | Porównanie 3 modeli (test + CV) |
| `cv_scores.json` | Wyniki 5-fold CV dla RF (per fold + średnia ± odchylenie) |
| `preprocessing_info.json` | Lista usuniętych kolumn, zakodowanych, skalowanych, finalne cechy |
| `eda_summary.json` | Statystyki EDA, mutual information |
| `roc_data.json` | Dane do wykresu ROC (per klasa) |
| `feature_importance.json` | Ranking ważności cech RF |
| `confusion_matrix.npy` | Macierz pomyłek (tablica NumPy) |
| `*.png` | 10 wykresów (rozkład klas, braki, dystrybucje, korelacje, MI, porównanie modeli, macierz pomyłek, ROC, ważność cech) |
| `artifacts.json` | Spis wszystkich wygenerowanych plików |

Wytrenowany las losowy zapisuje się jako [models/random_forest.joblib](models/) i może zostać wczytany w przyszłości przez `joblib.load(...)`.

---

## 8. Podsumowanie

Projekt realizuje **kompletny pipeline ML** dla problemu klasyfikacji wieloklasowej ataków sieciowych: od pobrania danych, przez EDA, preprocessing, trening trzech modeli z walidacją krzyżową, aż po szczegółową ewaluację na zbiorze testowym i wizualizację w dashboardzie Streamlit.

Mimo że metryki predykcji są w tym konkretnym zbiorze niskie (≈ losowe zgadywanie), **metodologia jest prawidłowa**, a wnioski uczciwe — to ważna lekcja: w prawdziwej pracy z danymi sam wybór modelu nie zastąpi sensownej zawartości informacyjnej cech.
