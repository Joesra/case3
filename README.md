# Case 3 – Vluchten en vertraging op Zürich Airport

Streamlit-dashboard bij de vraag: **wat voorspelt vertraging op Zürich Airport?**

## Starten

```
pip install -r requirements.txt
streamlit run app.py
```

Er zijn geen handmatige stappen: de app leest de drie bronnen uit `data/`, schoont ze op en combineert ze bij het opstarten.

## Bestanden

| Bestand | Wat het doet |
|---|---|
| `app.py` | Het dashboard (zeven pagina's, filters in de zijbalk) |
| `data_voorbereiden.py` | Inlezen, inspecteren, opschonen, luchthavens en weer koppelen, vertraging berekenen |
| `model.py` | Voorspelmodel: kans op 15 minuten of meer vertraging, en verwachte minuten |
| `start_dashboard.py` | Start het dashboard vanuit VS Code met de Run-knop |
| `data/schedule_airport.zip` | 323.461 vluchten van en naar Zürich, 2019–2020 (Brightspace) |
| `data/airports-extended.csv` | Luchthavens met IATA/ICAO en coördinaten (OpenFlights, via Kaggle) |
| `data/weer_zurich_2019_2020.csv` | Dagweer station 06670 Zürich-Kloten (Meteostat) |

## Bronnen data

- Vluchten: `schedule_airport.csv`, aangeleverd via Brightspace.
- Luchthavens: https://www.kaggle.com/datasets/open-flights/airports-train-stations-and-ferry-terminals
- Weer: https://bulk.meteostat.net/v2/daily/06670.csv.gz (station 06670, Zürich-Kloten), de dagen van 2019 en 2020.
  Het bulkbestand heeft geen kopregel; de kolomnamen komen uit https://dev.meteostat.net/bulk/daily.html

## Bronnen code

Overgenomen ideeën en voorbeelden staan ook als commentaar bij de code zelf.

- Streamlit caching: https://docs.streamlit.io/develop/concepts/architecture/caching
- Plotly kaarten en lijnen op kaarten: https://plotly.com/python/scatter-plots-on-maps/ en https://plotly.com/python/lines-on-maps/
- Haversine-formule voor afstand: https://en.wikipedia.org/wiki/Haversine_formula
- Gradient boosting: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- Kalibratie van kansen: https://scikit-learn.org/stable/modules/calibration.html

## Publiceren

1. Zet deze map in een GitHub-repository (alles behalve wat in `.gitignore` staat).
2. Ga naar https://share.streamlit.io, kies de repository en `app.py` als hoofdbestand.
3. De link die je krijgt is de link die je inlevert.
