# Maciej Romanski 
Praca magisterska w ramach projektu LOB:\
*Krótkoterminowe predykcje prędkości wiatru przy użyciu sztucznych sieci neuronowych*


## Środowisko wirtualne
Zainstaluj pakiet [pipenv](https://pipenv.pypa.io/en/latest/):
```
pip install pipenv
```
Za jego pomocą stwórz środowisko wirtualne z potrzebnymi pakietami:
```
pipenv install
```
Aktywuj stworzone środowisko:
```
pipenv shell
```

## Pobieranie danych
Używanie dane można znaleźć na dysku:\
https://drive.google.com/drive/u/0/folders/1OApT-cvotMwzyEM3yf_JeoLLtX67NQGG

Można je również pobrać bezpośrednio z bazy ERA5 poprzez [cdsapi](https://cds.climate.copernicus.eu/api-how-to). Stworzono do tego celu skrypt `data_downloader.py`. Wysyła on oddzielny request dla każdego roku, co pozwala ominąć limit wielkości pobranych danych, ale zajmuje dużo czasu.

Dane (pliki z rozszerzeniem .nc) należy umieścić w folderze *'data/ERA5_single_location/'*.

## Trening i ewaluacja modelu
Skrypt `models.py` wczytuje dane, przygotowuje je, trenuje prosty model gęstej sieci neuronowej, oblicza błąd RMSE oraz zapisuje wykres z przykładowymi wynikami w katalogu *'workdir/figures/'*.

## Notatniki
W katalogu *notebooks* umieszczono notatniki Jupyter, używane do kolejnych badań.