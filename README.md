# master_thesis
master thesis at ISZ AiR AGH

### **Instrukcja uruchamiania programu do klasteryzacji plikow tekstowych z selekcją cech przy użyciu algorytmu PSO** ### 

##### **Pliki:** #####
- document.py
- corpus.py
- particle.py
- pso.py
- clustering.py
- main.py

**Plik document.py:**
- zawiera klase Document przechowującą informacje o wczytanym dokumencie tekstowym
- uruchamianie funkcji preprocessingowych, takich jak: lemmatize(), stemm(), remove_non_alphanumeric(), tokenize(), remove_stop_words()

**Plik corpus.py:**
- zawiera klase przechowującą listę dokumentów 
- Wczytywanie iteracyjne plików z danego folderu
- uruchamianie funkcji potrzebnych do obliczenia TF-IDF, czyli: get_vocabulary(), tf_idf()

**Plik particle.py:**
- Zawiera klasę Particle reprezentującą jedną cząstkę w algorytmie PSO
- funkcje takie jak: update_velocity(), update_position(), evaluate(), cost_function()

**Plik pso.py:**
- Zawiera klasę PSO przechowującą listę cząstek (particles)
- inicjalizacja roju
- uruchomienie działania algorytmu PSO

**Plik clustering.py:**
- uruchomienie klasteryzacji plików tesktowych
- inicjalizacja centroidów
- walidacja klasteryzacji


### **Uruchomienie programu** ###
Piliki tekstowe z instancji testowej powinny znajdować się w lokalizacji pliku main.py. Program uruchamia się poprzez uruchomienie pliku main.py komendą: 
>> `python main.py`

##### **Parametry:** #####
> * pso_parameters (dict[str: List[int]]):
        - max_iter, domyslnie ustawione na [10, 20, 50, 100] <br>
        - num_features_select, domyslnie ustawione na [1, 2, 3, 5, 10, 20]<br>
        - num_particles, domyslnie ustawione na [10, 20, 50, 100]<br>
        - w, domyslnie ustawione na [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]<br>
        - c1, domyslnie ustawione na [1.5, 1.6, 1.7, 1.8, 1.9, 2]<br>
        - c2, domyslnie ustawione na [1.5, 1.6, 1.7, 1.8, 1.9, 2]<br>
> * k_means_parameters (dict[str: List[int]]):
        - k_means_max_iter, domyslnie ustawione na [5, 10, 20, 50]<br>
        - k_means_n_init, domyslnie ustawione na [10, 20, 50, 100, 200]<br>
> * selected_key (str)- nazwa parametru dla którego będzie znajdowana najlepsza wartość, przy pozostałych parametrach ustawionych na stałą wartość
> * select_features (bool)- jeśli wartość ustalona na `true` wtedy cechy będą wyselekcjonowane przy użyciu algorytmu PSO, jeśli na `false` wtedy selekcja cech nie będzie przeprowadzona
> * instance_folder (str)- nazwa folderu do którego będą zapisywane wyniki (powinien znajdować się w folderze /results)
        
Zmiana parametrów możliwa poprzez zmiane/dodanie/usunięcie wartości z list słowników: pso_parameters i k_means_parameters znajdujących się w pliku main.py.

Po uruchomieniu programu wyniki testów będą wpisywanie do plików tekstowych w folderze /results/<instance_folder>.
