"""
Case 3 - Voorspellen van vertraging.

Twee modellen op dezelfde kenmerken:
    - een classifier: de kans dat een vlucht 15 minuten of meer vertraging heeft
    - een regressor:  het verwachte aantal minuten vertraging

Gebruikte techniek: histogram-based gradient boosting uit scikit-learn.
Bron: https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
Die kan zelf omgaan met ontbrekende waarden en met categorieen, en traint
snel genoeg om live in het dashboard te draaien.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# De kenmerken per bron. De volgorde is de volgorde waarin we ze toevoegen,
# zodat je ziet wat elke bron extra oplevert.
KENMERKEN_PER_BRON = {
    "Rooster": ["uur", "weekdag", "maand", "drukte", "vluchten_dag",
                "richting", "ACT", "RWY", "maatschappij"],
    "+ Luchthavens": ["afstand_km", "Latitude", "Longitude"],
    "+ Weer": ["tavg", "tmin", "prcp", "wspd", "wpgt", "pres"],
    "+ Vorig uur": ["vertraging_vorig_uur"],
}
CATEGORISCH = ["richting", "ACT", "RWY", "maatschappij"]
MAX_CATEGORIEEN = 60          # zeldzame toestellen/maatschappijen gaan samen in 'Overig'

NAMEN = {
    "uur": "Uur van de dag", "weekdag": "Dag van de week", "maand": "Maand",
    "drukte": "Vluchten in hetzelfde uur", "vluchten_dag": "Vluchten op die dag",
    "richting": "Vertrek of aankomst", "ACT": "Vliegtuigtype", "RWY": "Baan",
    "maatschappij": "Maatschappij", "afstand_km": "Afstand", "Latitude": "Breedtegraad bestemming",
    "Longitude": "Lengtegraad bestemming", "tavg": "Temperatuur", "tmin": "Minimumtemperatuur",
    "prcp": "Neerslag", "wspd": "Windsnelheid", "wpgt": "Windstoten", "pres": "Luchtdruk",
    "vertraging_vorig_uur": "Vertraging in het uur ervoor",
}

TESTMETHODEN = {
    "Willekeurige dagen (80% trainen, 20% testen)": "dagen",
    "Trainen op 2019, testen op 2020": "jaar",
}


def voeg_modelkenmerken_toe(df: pd.DataFrame) -> pd.DataFrame:
    """Twee extra kenmerken die we uit het rooster zelf afleiden."""
    df = df.copy()
    df["vluchten_dag"] = df.groupby("datum")["FLT"].transform("size")

    # Gemiddelde vertraging van alle vluchten die het uur ervoor gepland stonden.
    # Dit is informatie die je pas op de dag zelf hebt, niet dagen vooruit.
    per_uur = df.groupby(["datum", "uur"])["vertraging_min"].mean().reset_index()
    per_uur["uur"] = per_uur["uur"] + 1
    per_uur = per_uur.rename(columns={"vertraging_min": "vertraging_vorig_uur"})
    df = df.merge(per_uur, how="left", on=["datum", "uur"])
    return df


def bepaal_categorieen(df: pd.DataFrame) -> dict:
    """Per categorische kolom de meest voorkomende waarden, de rest wordt 'Overig'."""
    return {
        kolom: list(df[kolom].value_counts().head(MAX_CATEGORIEEN).index.astype(str)) + ["Overig"]
        for kolom in CATEGORISCH
    }


def codeer(df: pd.DataFrame, kolommen: list, categorieen: dict) -> np.ndarray:
    """Maakt van de gekozen kolommen een getallenmatrix voor het model."""
    X = pd.DataFrame(index=df.index)
    for kolom in kolommen:
        if kolom in CATEGORISCH:
            waarden = df[kolom].astype(str)
            waarden = waarden.where(waarden.isin(categorieen[kolom]), "Overig")
            X[kolom] = pd.Categorical(waarden, categories=categorieen[kolom]).codes
        else:
            X[kolom] = df[kolom].astype(float)
    return X.to_numpy(dtype=float)


def _splits(df: pd.DataFrame, methode: str):
    """Train- en testrijen. We splitsen op hele dagen: alle vluchten van een dag
    delen hetzelfde weer, dus een dag half in train en half in test zou valsspelen."""
    rijen = np.arange(len(df))
    if methode == "jaar":
        return rijen[(df["jaar"] == 2019).to_numpy()], rijen[(df["jaar"] == 2020).to_numpy()]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(splitter.split(rijen, groups=df["datum"]))
    return train, test


def train_modellen(df: pd.DataFrame, methode: str = "dagen") -> dict:
    """Traint de modellen en geeft alles terug wat het dashboard wil laten zien."""
    df = voeg_modelkenmerken_toe(df)
    categorieen = bepaal_categorieen(df)
    train, test = _splits(df, methode)

    y_klasse = df["vertraagd"].to_numpy().astype(int)
    # Voor het minutenmodel kappen we extreme waarden af bij het trainen, anders
    # trekt een handvol vluchten met 10 uur vertraging het hele model scheef.
    y_min = df["vertraging_min"].to_numpy()
    y_min_train = np.clip(y_min, -60, 180)

    # --- Basislijn: zonder model ------------------------------------------
    aandeel_train = y_klasse[train].mean()
    basis = {
        "aandeel_vertraagd_train": float(aandeel_train),
        "aandeel_vertraagd_test": float(y_klasse[test].mean()),
        "brier": float(brier_score_loss(y_klasse[test], np.full(len(test), aandeel_train))),
        "mae": float(mean_absolute_error(y_min[test], np.full(len(test), np.median(y_min[train])))),
    }

    # --- Stap voor stap een bron erbij ------------------------------------
    stappen, kolommen = [], []
    classifier = None
    for stap, nieuwe_kolommen in KENMERKEN_PER_BRON.items():
        kolommen = kolommen + nieuwe_kolommen
        X = codeer(df, kolommen, categorieen)
        cat_index = [i for i, kolom in enumerate(kolommen) if kolom in CATEGORISCH]
        classifier = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.1, categorical_features=cat_index, random_state=42,
        ).fit(X[train], y_klasse[train])
        kans = classifier.predict_proba(X[test])[:, 1]
        stappen.append({
            "Stap": stap,
            "AUC": roc_auc_score(y_klasse[test], kans),
            "Brier": brier_score_loss(y_klasse[test], kans),
            "Aantal kenmerken": len(kolommen),
        })
    ladder = pd.DataFrame(stappen)

    # --- Minutenmodel op alle kenmerken -----------------------------------
    regressor = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.1, categorical_features=cat_index, random_state=42,
    ).fit(X[train], y_min_train[train])
    verwacht = regressor.predict(X[test])

    # --- Welke kenmerken doen ertoe? --------------------------------------
    # Permutation importance: hussel een kolom en kijk hoeveel slechter het model wordt.
    # Bron: https://scikit-learn.org/stable/modules/permutation_importance.html
    rng = np.random.RandomState(42)
    steekproef = rng.choice(test, size=min(8000, len(test)), replace=False)
    belang = permutation_importance(
        classifier, X[steekproef], y_klasse[steekproef],
        scoring="roc_auc", n_repeats=2, random_state=42,
    )
    belang = (
        pd.DataFrame({"Kenmerk": [NAMEN[k] for k in kolommen], "Daling AUC": belang.importances_mean})
        .sort_values("Daling AUC", ascending=False).reset_index(drop=True)
    )

    # --- Testset met voorspellingen erbij, voor de foutenanalyse -----------
    testset = df.iloc[test][
        ["datum", "jaar", "uur", "richting", "drukte", "RWY", "wpgt", "prcp", "tavg",
         "vertraging_min", "vertraagd", "vertraging_vorig_uur"]
    ].copy()
    testset["kans"] = kans
    testset["verwacht_min"] = verwacht

    return {
        "methode": methode,
        "n_train": len(train), "n_test": len(test),
        "dagen_train": int(df["datum"].iloc[train].nunique()),
        "dagen_test": int(df["datum"].iloc[test].nunique()),
        "basis": basis,
        "ladder": ladder,
        "mae": float(mean_absolute_error(y_min[test], verwacht)),
        "r2": float(r2_score(y_min[test], verwacht)),
        "belang": belang,
        "testset": testset,
        "classifier": classifier, "regressor": regressor,
        "kolommen": kolommen, "categorieen": categorieen,
    }


def voorspel_een_vlucht(resultaat: dict, vlucht: dict) -> tuple[float, float]:
    """Kans op vertraging en verwachte minuten voor een zelf ingevulde vlucht."""
    rij = pd.DataFrame([vlucht])
    X = codeer(rij, resultaat["kolommen"], resultaat["categorieen"])
    kans = float(resultaat["classifier"].predict_proba(X)[0, 1])
    minuten = float(resultaat["regressor"].predict(X)[0])
    return kans, minuten


def kalibratie(testset: pd.DataFrame, groepen: int = 10) -> pd.DataFrame:
    """Verdeelt de testvluchten in groepen van oplopende voorspelde kans en
    vergelijkt per groep de voorspelde kans met wat er echt gebeurde.
    Idee: https://scikit-learn.org/stable/modules/calibration.html"""
    groep = pd.qcut(testset["kans"], groepen, duplicates="drop")
    return (
        testset.groupby(groep, observed=True)
        .agg(voorspeld=("kans", "mean"), werkelijk=("vertraagd", "mean"), vluchten=("kans", "size"))
        .reset_index(drop=True)
    )


def verwarring(testset: pd.DataFrame, drempel: float) -> dict:
    """Telt goed en fout bij een gekozen drempel voor 'het model zegt: vertraagd'."""
    voorspeld = testset["kans"] >= drempel
    echt = testset["vertraagd"]
    terecht_alarm = int((voorspeld & echt).sum())
    vals_alarm = int((voorspeld & ~echt).sum())
    gemist = int((~voorspeld & echt).sum())
    terecht_rustig = int((~voorspeld & ~echt).sum())
    return {
        "terecht_alarm": terecht_alarm, "vals_alarm": vals_alarm,
        "gemist": gemist, "terecht_rustig": terecht_rustig,
        "precisie": terecht_alarm / max(terecht_alarm + vals_alarm, 1),
        "vangst": terecht_alarm / max(terecht_alarm + gemist, 1),
        "juist": (terecht_alarm + terecht_rustig) / len(testset),
    }
