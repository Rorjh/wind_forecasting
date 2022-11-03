# Maciej Romanski 
Praca magisterska w ramach projektu LOB:\
*Krótkoterminowe predykcje prędkości wiatru przy użyciu sztucznych sieci neuronowych*


## Środowisko wirtualne
Stwórz środowisko wirtualne pythona:
```
virtualvenv venv
```
Zainstaluj potrzebne pakiety z pliku requirements.txt:
```
pip install -r requirements.txt
```

<!-- Zainstaluj pakiet [pipenv](https://pipenv.pypa.io/en/latest/):
```
pip install pipenv
```
Za jego pomocą stwórz środowisko wirtualne z potrzebnymi pakietami (z pliku requirements.py):
```
pipenv install
```
Aktywuj stworzone środowisko:
```
pipenv shell
``` -->
## Uzupełnij plik konfiguracyjny
W pliku *properties.cfg* należy dostosować ustawienia.
```
[dataSection]
latitude = 50
longitude = 20
train.from = 2013
train.to = 2020
test.from = 2021
test.to = 2021

[modelsSection]
lookback = 24
noHoursPredicted = 6
dense = True
lstm = False
```
* *lookback* - ile kroków czasowych historii jest używane do wykonania predykcji
* *noHoursPredicted* - na ile godzin do przodu wykonywana jest predykcja
* *dense, lstm* - określają, czy poszczególny rodzaj modelu ma zostać wytrenowany

## Pobieranie danych
Używanie dane można znaleźć na dysku:\
https://drive.google.com/drive/u/0/folders/1OApT-cvotMwzyEM3yf_JeoLLtX67NQGG

Należy je (pliki z rozszerzeniem .nc) umieścić w folderze *'data/'*.

Można je również pobrać bezpośrednio z bazy ERA5 poprzez [cdsapi](https://cds.climate.copernicus.eu/api-how-to). Stworzono do tego celu skrypt `data_downloader.py`. Wysyła on oddzielny request dla każdego roku, co pozwala ominąć limit wielkości pobranych danych, ale zajmuje dużo czasu. Aby skorzystać z cdsapi, należy najpierw założyć konto i skonfigurować plik z kluczem, zgodnie z [instrukcją na stronie](https://cds.climate.copernicus.eu/api-how-to)

Aby pobrać za pomocą skryptu dane dla lokalizacji i lat określonych w pliku konfiguracyjnym (zarówno dla okresu treningowego jak i testowego):
```
python data_downloader.py
```
Dane są zapisywane w folderze *data*.

## Trening i ewaluacja modelu
Skrypt `models.py` wczytuje dane, przygotowuje je i trenuje na nich dwa modele (w zależności od wybranych w pliku konfiguracyjnym): prosty model gęstej sieci neuronowej oraz bardziej złożony model oparty na warstwach LSTM. Następnie skrypt oblicza błąd RMSE dla okresu testowego, zapisuje wykresy z przykładowymi wynikami w katalogu *figures* oraz zapisuje modele do plików *.h5*, aby można było wykonać przy ich użyciu predykcje w przyszłości.

## Wykonanie predykcji na podanych danych:
Aby wykonać przewidywania na rzeczywistych danych można skorzystać ze skryptu *make_predictions.py*. Należy w tym celu zapisać dane wejściowe w pliku .csv w formacie zgodnym z przykładem znajdującym się w katalogu *templates*. Liczba kroków czasowych zawartych w pliku .csv musi być równa parametrowi *lookback* z pliku konfiguracyjnego. Skrypt należy uruchomić komendą:
```
python make_predictions.py pathToModel pathToInputFile
```
gdzie: 
* *pathToModel* - ścieżka do pliku .h5 z zapisanym modelem
* *pathToInputFile* - ścieżka do pliku .csv z danymi wejściowymi

## Notatniki
W katalogu *notebooks* umieszczono notatniki Jupyter, używane do kolejnych badań.