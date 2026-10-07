"""
Case 3 - Vluchten en vertraging op Zurich Airport
Inlezen, inspecteren, opschonen en combineren van de drie bronnen.

Bronnen (staan in de map data/):
    schedule_airport.zip         vluchten van/naar Zurich, 2019-2020 (Brightspace)
    airports-extended.csv        luchthavens met IATA/ICAO en coordinaten (OpenFlights, via Kaggle)
    weer_zurich_2019_2020.csv    dagweer station 06670 Zurich-Kloten (Meteostat, bulk.meteostat.net)

Dit bestand bevat geen Streamlit-code, zodat je het ook los kunt draaien:
    python data_voorbereiden.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_MAP = Path(__file__).parent / "data"

# Kolomnamen van het OpenFlights-bestand (het bestand zelf heeft geen kopregel).
# Bron kolomvolgorde: https://openflights.org/data.php
AIRPORT_KOLOMMEN = [
    "Airport_ID", "Name", "City", "Country", "IATA", "ICAO",
    "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz", "Type", "Source",
]

# Kolomnamen van het Meteostat-dagbestand.
# Bron: https://dev.meteostat.net/bulk/daily.html
WEER_KOLOMMEN = {
    "tavg": "Gemiddelde temperatuur (°C)",
    "tmin": "Minimumtemperatuur (°C)",
    "tmax": "Maximumtemperatuur (°C)",
    "prcp": "Neerslag (mm)",
    "wspd": "Gemiddelde windsnelheid (km/u)",
    "wpgt": "Zwaarste windstoot (km/u)",
    "pres": "Luchtdruk (hPa)",
}

ZURICH_ICAO = "LSZH"
VERTRAAGD_VANAF_MIN = 15      # gangbare grens in de luchtvaart: 15 minuten of meer = vertraagd
LANGE_AFSTAND_KM = 3500       # grens tussen Europa/omgeving en intercontinentaal

WEEKDAGEN = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag", "Zaterdag", "Zondag"]
MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


# --------------------------------------------------------------------------
# 1. Inlezen
# --------------------------------------------------------------------------
def laad_ruwe_vluchten(map_in: Path = DATA_MAP) -> pd.DataFrame:
    zip_pad = map_in / "schedule_airport.zip"
    csv_pad = map_in / "schedule_airport.csv"

    pad = zip_pad if zip_pad.exists() else csv_pad

    return pd.read_csv(
        pad,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        compression="zip" if pad.suffix == ".zip" else None
    )


def laad_luchthavens(map_in: Path = DATA_MAP) -> pd.DataFrame:
    """Luchthavenbestand inlezen en schoonmaken."""
    df = pd.read_csv(
        map_in / "airports-extended.csv", header=None, names=AIRPORT_KOLOMMEN,
        dtype=str, keep_default_na=False, encoding="utf-8",
    )
    df = df.apply(lambda kolom: kolom.str.strip())
    # '\N' is in OpenFlights de code voor een ontbrekende waarde
    df = df.replace({"\\N": pd.NA, "": pd.NA})
    for kolom in ["Latitude", "Longitude", "Altitude", "Timezone"]:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")
    return df


def laad_weer(map_in: Path = DATA_MAP) -> tuple[pd.DataFrame, dict]:
    """
    Dagweer van Zurich-Kloten inlezen en controleren.

    Geeft het schone weer terug plus een rapportje met wat er mis was.
    """
    weer = pd.read_csv(map_in / "weer_zurich_2019_2020.csv", parse_dates=["date"])
    rapport = {
        "dagen": len(weer),
        "eerste_dag": weer["date"].min(),
        "laatste_dag": weer["date"].max(),
        "leeg_per_kolom": weer.isna().sum().to_dict(),
    }

    # Sneeuw en zonuren zijn in deze twee jaar bij de bron helemaal leeg: weg ermee.
    weer = weer.drop(columns=["snow", "tsun"])

    # Vanaf 11 november 2020 staat er bijna elke dag ~21,6 mm neerslag.
    # 51 dagen achter elkaar (bijna) dezelfde waarde is geen weer maar een
    # opvulwaarde van de bron. We zetten die neerslag op 'onbekend'.
    verdacht = weer["date"] >= "2020-11-11"
    rapport["neerslag_verdacht_dagen"] = int(verdacht.sum())
    rapport["neerslag_verdacht_gemiddeld"] = float(weer.loc[verdacht, "prcp"].mean())
    rapport["neerslag_normaal_gemiddeld"] = float(weer.loc[~verdacht, "prcp"].mean())
    weer.loc[verdacht, "prcp"] = np.nan

    weer = weer.rename(columns={"date": "datum"})
    return weer, rapport


# --------------------------------------------------------------------------
# 2. Inspecteren
# --------------------------------------------------------------------------
def inspecteer(ruw: pd.DataFrame) -> pd.DataFrame:
    """Per kolom: hoeveel unieke waarden, hoeveel 'lege' tekens en een voorbeeld."""
    rijen = []
    for kolom in ruw.columns:
        waarden = ruw[kolom]
        rijen.append({
            "Kolom": kolom,
            "Unieke waarden": waarden.nunique(),
            "Aantal '-'": int((waarden == "-").sum()),
            "Aantal '#N/A'": int((waarden == "#N/A").sum()),
            "Leeg (%)": round(100 * waarden.isin(["-", "#N/A", ""]).mean(), 1),
            "Voorbeeld": waarden.iloc[0],
        })
    return pd.DataFrame(rijen)


# --------------------------------------------------------------------------
# 3. Opschonen en vertraging berekenen
# --------------------------------------------------------------------------
def schoon_vluchten(ruw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Vluchten opschonen en de vertragingskolom maken."""
    rapport = {"rijen_ruw": len(ruw), "kolommen_ruw": ruw.shape[1]}
    df = ruw.apply(lambda kolom: kolom.str.strip())

    # '-' en '#N/A' betekenen in dit bestand "geen waarde"
    df = df.replace({"-": pd.NA, "#N/A": pd.NA, "": pd.NA})

    # Exact dubbele rijen verwijderen
    voor = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    rapport["dubbele_rijen"] = voor - len(df)

    # Zelfde vlucht (datum + tijd + vluchtnummer) die twee keer voorkomt met een
    # andere werkelijke tijd. We weten niet welke klopt, dus beide blijven staan.
    rapport["dubbele_identifier"] = int(df["Identifier"].duplicated().sum())

    # --- Datum en tijd -----------------------------------------------------
    datum = pd.to_datetime(df["STD"], format="%d/%m/%Y")
    df["gepland"] = datum + pd.to_timedelta(df["STA_STD_ltc"])
    df["werkelijk"] = datum + pd.to_timedelta(df["ATA_ATD_ltc"])

    # Het bestand bevat alleen de geplande datum. Een vlucht die om 22:45 gepland
    # stond en om 00:05 vertrok, vertrok de volgende dag. Zonder correctie lijkt
    # die vlucht bijna 23 uur te vroeg. Ligt de werkelijke tijd meer dan 12 uur
    # voor de geplande tijd, dan tellen we er een dag bij op.
    ruwe_vertraging = (df["werkelijk"] - df["gepland"]).dt.total_seconds() / 60
    over_middernacht = ruwe_vertraging < -12 * 60
    df.loc[over_middernacht, "werkelijk"] += pd.Timedelta(days=1)
    df["over_middernacht"] = over_middernacht
    rapport["over_middernacht"] = int(over_middernacht.sum())
    rapport["minimum_voor_correctie"] = float(ruwe_vertraging.min())

    # --- De vertragingskolom -----------------------------------------------
    df["vertraging_min"] = (df["werkelijk"] - df["gepland"]).dt.total_seconds() / 60
    df["vertraagd"] = df["vertraging_min"] >= VERTRAAGD_VANAF_MIN
    rapport["minimum_na_correctie"] = float(df["vertraging_min"].min())
    rapport["maximum_na_correctie"] = float(df["vertraging_min"].max())

    # --- Afgeleide kolommen voor de analyse --------------------------------
    df["datum"] = datum
    df["jaar"] = datum.dt.year
    df["maand"] = datum.dt.month
    df["weekdag"] = datum.dt.dayofweek            # 0 = maandag
    df["uur"] = df["gepland"].dt.hour
    df["richting"] = df["LSV"].map({"S": "Vertrek", "L": "Aankomst"})

    # Maatschappij uit het vluchtnummer: is het derde teken een letter, dan is
    # het een drieletterige code (EZY2191), anders een tweetekencode (LX092, A3850).
    derde_is_letter = df["FLT"].str[2].str.isalpha()
    df["maatschappij"] = np.where(derde_is_letter, df["FLT"].str[:3], df["FLT"].str[:2])

    # Drukte: hoeveel vluchten (start + landing) staan er in hetzelfde uur gepland?
    df["drukte"] = df.groupby(["datum", "uur"])["FLT"].transform("size")

    return df, rapport


# --------------------------------------------------------------------------
# 4. Combineren met luchthavens en weer
# --------------------------------------------------------------------------
def afstand_km(lat1, lon1, lat2, lon2):
    """
    Afstand over de aardbol (grootcirkel) met de haversine-formule.
    Bron formule: https://en.wikipedia.org/wiki/Haversine_formula
    """
    straal_aarde = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * straal_aarde * np.arcsin(np.sqrt(a))


def koppel_luchthavens(vluchten: pd.DataFrame, luchthavens: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Vluchten koppelen aan luchthavens via de code in 'Org/Des'.

    'Org/Des' bevat bijna altijd een ICAO-code (4 letters, bv. LFPG), maar bij
    een paar honderd vluchten een IATA-code (3 letters, bv. BER). Daarom eerst
    koppelen op ICAO en de rest aanvullen via IATA.
    """
    info = luchthavens[["Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Type"]]
    op_icao = info.dropna(subset=["ICAO"]).drop_duplicates("ICAO")
    op_iata = info.dropna(subset=["IATA"]).drop_duplicates("IATA")

    codes = vluchten[["Org/Des"]]
    via_icao = codes.merge(op_icao, how="left", left_on="Org/Des", right_on="ICAO")
    via_iata = codes.merge(op_iata, how="left", left_on="Org/Des", right_on="IATA")
    assert len(via_icao) == len(vluchten) == len(via_iata), "merge heeft rijen toegevoegd"

    gevonden_icao = via_icao["Name"].notna()
    gevonden_iata = ~gevonden_icao & via_iata["Name"].notna()
    # ICAO-match heeft voorrang, de IATA-match vult de gaten
    gevonden = via_icao.drop(columns="Org/Des").combine_first(via_iata.drop(columns="Org/Des"))
    df = pd.concat([vluchten, gevonden[info.columns]], axis=1)

    alle_codes = vluchten["Org/Des"].dropna().unique()
    rapport = {
        "bestemmingen": len(alle_codes),
        "bestemmingen_via_icao": int(pd.Series(alle_codes).isin(op_icao["ICAO"]).sum()),
        "bestemmingen_via_iata_kolom": int(pd.Series(alle_codes).isin(op_iata["IATA"]).sum()),
        "vluchten_via_icao": int(gevonden_icao.sum()),
        "vluchten_via_iata": int(gevonden_iata.sum()),
        "vluchten_zonder_code": int(vluchten["Org/Des"].isna().sum()),
        "vluchten_code_onbekend": int((vluchten["Org/Des"].notna() & df["Name"].isna()).sum()),
        "onbekende_codes": (
            vluchten.loc[vluchten["Org/Des"].notna() & df["Name"].isna(), "Org/Des"]
            .value_counts().to_dict()
        ),
    }

    # Afstand vanaf Zurich en een grove indeling voor de kaart
    zurich = luchthavens.loc[luchthavens["ICAO"] == ZURICH_ICAO].iloc[0]
    df["afstand_km"] = afstand_km(zurich["Latitude"], zurich["Longitude"], df["Latitude"], df["Longitude"])
    df["gebied"] = np.where(df["afstand_km"] >= LANGE_AFSTAND_KM, "Intercontinentaal", "Europa en omgeving")
    df.loc[df["afstand_km"].isna(), "gebied"] = pd.NA
    rapport["zurich"] = (float(zurich["Latitude"]), float(zurich["Longitude"]))
    return df, rapport


def koppel_weer(vluchten: pd.DataFrame, weer: pd.DataFrame) -> pd.DataFrame:
    """Elke vlucht krijgt het weer van zijn (geplande) dag."""
    df = vluchten.merge(weer, how="left", on="datum")
    assert len(df) == len(vluchten), "weer-merge heeft rijen toegevoegd"
    # Vorst is de beste aanwijzing die we hebben voor winterse omstandigheden
    # (ijs van het toestel halen), want de sneeuwkolom is leeg.
    df["vorst"] = df["tmin"] < 0
    return df


# --------------------------------------------------------------------------
# 5. Alles achter elkaar
# --------------------------------------------------------------------------
def bouw_dataset(map_in: Path = DATA_MAP) -> tuple[pd.DataFrame, dict]:
    """
    Leest de drie bronnen, schoont ze op en combineert ze.

    Geeft terug:
        df       een rij per vlucht, met luchthaven, afstand, weer en vertraging
        rapport  alles wat we onderweg zijn tegengekomen (voor de pagina Data-inspectie)
    """
    ruw = laad_ruwe_vluchten(map_in)
    luchthavens = laad_luchthavens(map_in)
    weer, rapport_weer = laad_weer(map_in)

    rapport = {
        "inspectie_ruw": inspecteer(ruw),
        "voorbeeld_ruw": ruw.head(8),
        "luchthavens_rijen": len(luchthavens),
        "luchthavens_type": luchthavens["Type"].value_counts().to_dict(),
        "weer": rapport_weer,
    }

    df, rapport_schoon = schoon_vluchten(ruw)
    df, rapport_koppel = koppel_luchthavens(df, luchthavens)
    df = koppel_weer(df, weer)
    rapport.update(rapport_schoon)
    rapport["koppeling"] = rapport_koppel
    rapport["weer_dagen_gekoppeld"] = int(df["tavg"].notna().sum())

    # Kolommen die we niet meer nodig hebben weglaten en tekst als categorie
    # opslaan: dat scheelt veel geheugen (belangrijk op Streamlit Cloud).
    df = df.drop(columns=["STD", "STA_STD_ltc", "ATA_ATD_ltc", "Identifier", "Type"])
    for kolom in ["FLT", "LSV", "TAR", "GAT", "DL1", "IX1", "DL2", "IX2", "ACT", "RWY", "RWC",
                  "Org/Des", "Name", "City", "Country", "IATA", "ICAO", "richting",
                  "maatschappij", "gebied"]:
        df[kolom] = df[kolom].astype("category")
    for kolom in ["jaar", "maand", "weekdag", "uur", "drukte"]:
        df[kolom] = df[kolom].astype("int16")

    rapport["rijen_schoon"] = len(df)
    rapport["kolommen_schoon"] = df.shape[1]
    rapport["leeg_na_opschonen"] = df.isna().sum()
    return df, rapport


if __name__ == "__main__":
    dataset, info = bouw_dataset()
    print(f"Rijen: {len(dataset):,}  kolommen: {dataset.shape[1]}")
    print(f"Over middernacht gecorrigeerd: {info['over_middernacht']}")
    print(f"Gekoppeld via ICAO: {info['koppeling']['vluchten_via_icao']:,}, "
          f"via IATA: {info['koppeling']['vluchten_via_iata']:,}, "
          f"niet gevonden: {info['koppeling']['vluchten_code_onbekend']}")
    print(f"Gemiddelde vertraging: {dataset['vertraging_min'].mean():.1f} min")
    print(f"Geheugen: {dataset.memory_usage(deep=True).sum() / 1e6:.0f} MB")
