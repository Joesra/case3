"""
Case 3 - Van data naar informatie
Dashboard: wat voorspelt vertraging op Zurich Airport?

Starten:  streamlit run app.py

Gebruikte documentatie (code is aangepast aan onze eigen data):
    Streamlit caching      https://docs.streamlit.io/develop/concepts/architecture/caching
    Streamlit widgets      https://docs.streamlit.io/develop/api-reference/widgets
    Plotly lijngrafieken   https://plotly.com/python/line-charts/
    Plotly kaarten         https://plotly.com/python/scatter-plots-on-maps/
    Plotly lijnen op kaart https://plotly.com/python/lines-on-maps/
    Plotly heatmap         https://plotly.com/python/heatmaps/
"""
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model as mdl
from data_voorbereiden import (
    DATA_MAP, LANGE_AFSTAND_KM, MAANDEN, VERTRAAGD_VANAF_MIN, WEEKDAGEN, WEER_KOLOMMEN, bouw_dataset,
)

st.set_page_config(page_title="Vertraging op Zürich Airport", page_icon="✈️", layout="wide")

# --------------------------------------------------------------------------
# Kleuren: elke kleur heeft een vaste betekenis in het hele dashboard
# --------------------------------------------------------------------------
BLAUW, ORANJE, GROEN, ROOD = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INKT, INKT_ZACHT, RASTER, LAND = "#0b0b0b", "#52514e", "#e6e5e1", "#eeede9"
KLEUR_RICHTING = {"Vertrek": BLAUW, "Aankomst": GROEN}
KLEUR_JAAR = {2019: INKT_ZACHT, 2020: ORANJE}          # 2019 = het normale jaar, 2020 valt op
BLAUW_LICHT, BLAUW_MIDDEN, BLAUW_DONKER = "#9ec5f4", "#3987e5", "#0d366b"
SCHAAL_BLAUW = [[0, "#cde2fb"], [0.5, BLAUW_MIDDEN], [1, BLAUW_DONKER]]
SCHAAL_ROOD_BLAUW = [[0, ROOD], [0.5, "#f0efec"], [1, BLAUW]]

LOCKDOWN = "2020-03-16"      # Zwitserland kondigt de 'buitengewone situatie' af

# Maten die je in meerdere grafieken kunt kiezen: (kolom, eenheid, decimalen)
MATEN = {
    "Aandeel vertraagd (%)": ("aandeel", "%", 1),
    "Gemiddelde vertraging (min)": ("gem", " min", 1),
    "Aantal vluchten": ("vluchten", "", 0),
}


# --------------------------------------------------------------------------
# Hulpfuncties
# --------------------------------------------------------------------------
def nl(getal, decimalen=0):
    """Getal met Nederlandse notatie: 323.461 en 8,3"""
    if getal is None or (isinstance(getal, float) and np.isnan(getal)):
        return "–"
    tekst = f"{getal:,.{decimalen}f}"
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")


def _volle_breedte(functie):
    """Streamlit heeft de naam van de breedte-optie veranderd; dit werkt in oud en nieuw."""
    parameters = inspect.signature(functie).parameters
    if "width" in parameters and parameters["width"].default == "stretch":
        return {}
    return {"use_container_width": True}


def toon(fig):
    """Plotly-figuur tonen in onze eigen opmaak."""
    st.plotly_chart(fig, theme=None, config={"displaylogo": False}, **_volle_breedte(st.plotly_chart))


def tabel(df, **opties):
    st.dataframe(df, hide_index=True, **_volle_breedte(st.dataframe), **opties)


def opmaak(fig, x_titel=None, y_titel=None, hoogte=380, legenda=True, marge_rechts=20):
    """Dezelfde rustige opmaak voor elke grafiek: dun raster, legenda bovenaan."""
    fig.update_layout(
        template="plotly_white", height=hoogte, separators=",.",
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color=INKT_ZACHT),
        margin=dict(l=10, r=marge_rechts, t=40 if legenda else 15, b=10),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=dict(text="")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font=dict(size=13)),
    )
    fig.update_xaxes(title_text=x_titel, gridcolor=RASTER, linecolor="#b5b4ae", zeroline=False)
    fig.update_yaxes(title_text=y_titel, gridcolor=RASTER, zerolinecolor="#b5b4ae", rangemode="tozero")
    return fig


def hoverdata(data, kolommen):
    """Extra kolommen voor de hovertekst, met tekst en getallen door elkaar."""
    return data[kolommen].astype(object).to_numpy()


def samenvatting(data, groep):
    """Aantal vluchten, gemiddelde, mediaan en aandeel vertraagd per groep."""
    uit = (
        data.groupby(groep, observed=True)
        .agg(vluchten=("vertraging_min", "size"), gem=("vertraging_min", "mean"),
             mediaan=("vertraging_min", "median"), aandeel=("vertraagd", "mean"))
        .reset_index()
    )
    uit["aandeel"] = uit["aandeel"] * 100
    return uit


def lockdown_lijn(fig, tekst="16 mrt 2020: lockdown"):
    """Verticale lijn op een datum-as. Los getekend als vorm + tekst."""
    fig.add_shape(type="line", x0=LOCKDOWN, x1=LOCKDOWN, yref="paper", y0=0, y1=1,
                  line=dict(color=INKT_ZACHT, width=1, dash="dot"))
    fig.add_annotation(x=LOCKDOWN, yref="paper", y=1, text=tekst, showarrow=False,
                       xanchor="left", xshift=6, yanchor="top", font=dict(size=12, color=INKT_ZACHT))


def eindlabel(fig, x, y, tekst, kleur, yshift=0):
    """Naam van de lijn direct aan het eind van de lijn zetten."""
    fig.add_annotation(x=x, y=y, text=tekst, showarrow=False, xanchor="left", xshift=8, yshift=yshift,
                       font=dict(size=12, color=INKT), bgcolor="rgba(255,255,255,0.7)")


def eindlabels(fig, labels):
    """Meerdere eindlabels (x, y, tekst, kleur). Liggen twee eindpunten dicht bij elkaar,
    dan schuift de bovenste iets omhoog en de onderste iets omlaag."""
    labels = sorted(labels, key=lambda label: label[1])
    hoogste = max(abs(label[1]) for label in labels) or 1
    for i, (x, y, tekst, kleur) in enumerate(labels):
        dichtbij = any(abs(y - ander[1]) < 0.07 * hoogste for j, ander in enumerate(labels) if j != i)
        schuif = 0
        if dichtbij:
            schuif = 8 if i == len(labels) - 1 else (-8 if i == 0 else 0)
        eindlabel(fig, x, y, tekst, kleur, yshift=schuif)


# --------------------------------------------------------------------------
# Data laden (een keer, daarna uit de cache)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Data inlezen, opschonen en combineren…")
def laad():
    return bouw_dataset()


@st.cache_data
def laad_ruw_weer():
    return pd.read_csv(DATA_MAP / "weer_zurich_2019_2020.csv", parse_dates=["date"])


@st.cache_resource(show_spinner="Model trainen op ruim 250.000 vluchten… (ongeveer een halve minuut)")
def getraind_model(methode):
    df, _ = laad()
    return mdl.train_modellen(df, methode)


df, rapport = laad()

# --------------------------------------------------------------------------
# Zijbalk: navigatie en filters
# --------------------------------------------------------------------------
PAGINAS = [
    "1 · Overzicht",
    "2 · Data-inspectie",
    "3 · Tijd en drukte",
    "4 · Bestemmingen",
    "5 · Weer",
    "6 · 2019 tegenover 2020",
    "7 · Voorspelling",
]
st.sidebar.title("✈️ Zürich Airport")
st.sidebar.caption("Wat voorspelt vertraging?")
pagina = st.sidebar.radio("Pagina", PAGINAS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
jaar_keuze = st.sidebar.radio("Jaar", ["2019", "2020", "Beide jaren"], horizontal=True,
                              help="2019 is een normaal jaar, 2020 het coronajaar. Een gemiddelde over beide zegt weinig.")
richting_keuze = st.sidebar.radio("Richting", ["Vertrek en aankomst", "Vertrek", "Aankomst"])

aantal_per_maatschappij = df["maatschappij"].value_counts()
maatschappij_keuze = st.sidebar.multiselect(
    "Maatschappij", list(aantal_per_maatschappij.index[:30]),
    format_func=lambda code: f"{code} ({nl(aantal_per_maatschappij[code])})",
    help="Leeg = alle maatschappijen. LX is Swiss.",
)
aantal_per_toestel = df["ACT"].value_counts()
toestel_keuze = st.sidebar.multiselect(
    "Vliegtuigtype", list(aantal_per_toestel.index[:30]),
    format_func=lambda code: f"{code} ({nl(aantal_per_toestel[code])})",
    help="Leeg = alle types.",
)

# Het filter zonder het jaar hebben we nodig om jaren met elkaar te vergelijken
masker = pd.Series(True, index=df.index)
if richting_keuze != "Vertrek en aankomst":
    masker &= df["richting"] == richting_keuze
if maatschappij_keuze:
    masker &= df["maatschappij"].isin(maatschappij_keuze)
if toestel_keuze:
    masker &= df["ACT"].isin(toestel_keuze)
zonder_jaar = df[masker]
selectie = zonder_jaar if jaar_keuze == "Beide jaren" else zonder_jaar[zonder_jaar["jaar"] == int(jaar_keuze)]

st.sidebar.caption(f"Selectie: **{nl(len(selectie))}** van {nl(len(df))} vluchten")
with st.sidebar.expander("Bronnen"):
    st.markdown(
        "- **Vluchten**: `schedule_airport.csv` (Brightspace)\n"
        "- **Luchthavens**: OpenFlights, via Kaggle\n"
        "- **Weer**: Meteostat, station 06670 Zürich-Kloten\n\n"
        f"Vertraagd = {VERTRAAGD_VANAF_MIN} minuten of meer later dan gepland."
    )


def selectie_regel(filters_gelden=True):
    """Bovenaan elke pagina: waar kijk je naar?"""
    if not filters_gelden:
        st.caption("Deze pagina gebruikt altijd alle vluchten uit 2019 en 2020; de filters in de zijbalk gelden hier niet.")
        return
    delen = [jaar_keuze, richting_keuze.lower()]
    if maatschappij_keuze:
        delen.append("maatschappij " + ", ".join(maatschappij_keuze))
    if toestel_keuze:
        delen.append("type " + ", ".join(toestel_keuze))
    st.caption("Selectie: " + " · ".join(delen) + f" · {nl(len(selectie))} vluchten")


def stop_als_leeg(data, minimum=50):
    if len(data) < minimum:
        st.warning("Met deze filters blijven er te weinig vluchten over. Maak de selectie in de zijbalk ruimer.")
        st.stop()


# ==========================================================================
# 1 · OVERZICHT
# ==========================================================================
def pagina_overzicht():
    st.title("Wat voorspelt vertraging op Zürich Airport?")
    st.markdown(
        "Ruim 323.000 vluchten uit 2019 en 2020, gecombineerd met de **luchthavens** waar ze vandaan "
        "komen of naartoe gaan en met het **weer** van die dag in Zürich. Geen van die bronnen laat op "
        "zichzelf zien waar vertraging vandaan komt; samen wel."
    )
    selectie_regel()
    stop_als_leeg(selectie)

    # --- Kerncijfers, met het verschil ten opzichte van het andere jaar ----
    ander = None
    if jaar_keuze != "Beide jaren":
        ander_jaar = 2020 if jaar_keuze == "2019" else 2019
        ander = zonder_jaar[zonder_jaar["jaar"] == ander_jaar]
        if len(ander) == 0:
            ander = None

    def verschil(nu, toen, eenheid="", decimalen=1):
        if ander is None:
            return None
        plus = "+" if nu - toen > 0 else ""
        return f"{plus}{nl(nu - toen, decimalen)}{eenheid} t.o.v. {ander_jaar}"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Vluchten", nl(len(selectie)),
              verschil(len(selectie), len(ander) if ander is not None else 0, "", 0), delta_color="off")
    k2.metric("Vertraagd (≥ 15 min)", nl(selectie["vertraagd"].mean() * 100, 1) + "%",
              verschil(selectie["vertraagd"].mean() * 100, ander["vertraagd"].mean() * 100 if ander is not None else 0, " %-punt"),
              delta_color="inverse")
    k3.metric("Gemiddelde vertraging", nl(selectie["vertraging_min"].mean(), 1) + " min",
              verschil(selectie["vertraging_min"].mean(), ander["vertraging_min"].mean() if ander is not None else 0, " min"),
              delta_color="inverse")
    k4.metric("Mediaan vertraging", nl(selectie["vertraging_min"].median(), 1) + " min",
              verschil(selectie["vertraging_min"].median(), ander["vertraging_min"].median() if ander is not None else 0, " min"),
              delta_color="inverse")
    k5.metric("Bestemmingen", nl(selectie["Org/Des"].nunique()),
              verschil(selectie["Org/Des"].nunique(), ander["Org/Des"].nunique() if ander is not None else 0, "", 0),
              delta_color="off")

    # --- De lijngrafiek over de tijd ---------------------------------------
    st.markdown("#### Vertraging door de tijd")
    c1, c2, c3 = st.columns([2, 1, 1])
    maat = c1.radio("Toon", list(MATEN), horizontal=True, key="ov_maat")
    periode = c2.radio("Per", ["Week", "Maand", "Dag"], horizontal=True, key="ov_periode")
    splitsen = c3.checkbox("Vertrek en aankomst apart", value=richting_keuze == "Vertrek en aankomst")
    kolom, eenheid, decimalen = MATEN[maat]

    freq = {"Dag": "D", "Week": "W-MON", "Maand": "MS"}[periode]
    # De tijdlijn laat altijd beide jaren zien: de breuk in maart 2020 is het verhaal.
    groepen = zonder_jaar.groupby("richting", observed=True) if splitsen else [("Alle vluchten", zonder_jaar)]
    fig = go.Figure()
    einden = []
    for naam, deel in groepen:
        reeks = (
            deel.groupby(pd.Grouper(key="datum", freq=freq, closed="left", label="left"))
            .agg(vluchten=("vertraging_min", "size"), gem=("vertraging_min", "mean"), aandeel=("vertraagd", "mean"))
        )
        reeks["aandeel"] *= 100
        reeks = reeks[reeks["vluchten"] > 0]
        kleur = KLEUR_RICHTING.get(str(naam), BLAUW)
        if periode == "Dag":
            # Dagen schommelen sterk: dunne lijn voor de dag, dikke voor het 7-daags gemiddelde
            fig.add_trace(go.Scatter(x=reeks.index, y=reeks[kolom], mode="lines", name=f"{naam} (per dag)",
                                     line=dict(color=kleur, width=1), opacity=0.35, hoverinfo="skip", showlegend=False))
            reeks[kolom] = reeks[kolom].rolling(7, center=True, min_periods=4).mean()
            naam = f"{naam} (7-daags gemiddelde)"
        fig.add_trace(go.Scatter(
            x=reeks.index, y=reeks[kolom], mode="lines", name=str(naam), line=dict(color=kleur, width=2),
            hovertemplate="%{y:,." + str(decimalen) + "f}" + eenheid,
        ))
        zichtbaar = reeks[kolom].dropna()
        if splitsen and len(zichtbaar):
            einden.append((zichtbaar.index[-1], zichtbaar.iloc[-1], str(naam).split(" (")[0], kleur))
    if einden:
        eindlabels(fig, einden)
    opmaak(fig, y_titel=maat, hoogte=400, marge_rechts=90 if einden else 20)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(hoverformat="%d-%m-%Y", dtick="M3", tickformat="%m-%Y")
    lockdown_lijn(fig)
    toon(fig)
    st.caption("De grafiek toont altijd beide jaren, ook als in de zijbalk één jaar is gekozen; de overige filters gelden wel. "
               "Kies 'Dag' voor het 7-daags gemiddelde of 'Aantal vluchten' om de terugval zelf te zien.")

    # --- Belangrijkste bevindingen (berekend op alle data, niet op de filters)
    st.markdown("#### De vier belangrijkste bevindingen")
    st.caption("Berekend op alle vluchten, zodat ze niet meeschuiven met de filters.")
    v19 = df[(df["jaar"] == 2019) & (df["richting"] == "Vertrek")]
    druk, rustig = v19[v19["drukte"] > 60]["vertraagd"].mean() * 100, v19[v19["drukte"] <= 40]["vertraagd"].mean() * 100
    d19 = df[df["jaar"] == 2019]
    per_dag = d19.groupby("datum").agg(aandeel=("vertraagd", "mean"), wpgt=("wpgt", "first"))
    storm, kalm = per_dag[per_dag["wpgt"] >= 60]["aandeel"].mean() * 100, per_dag[per_dag["wpgt"] < 30]["aandeel"].mean() * 100
    na = df[df["maand"] >= 4].groupby("jaar").agg(n=("vertraagd", "size"), aandeel=("vertraagd", "mean"))
    b1, b2 = st.columns(2)
    b1.info(f"**Drukte jaagt vertraging aan.** In uren met meer dan 60 geplande vluchten vertrekt "
            f"{nl(druk)}% te laat, in uren met hooguit 40 vluchten {nl(rustig)}% (2019). → pagina 3")
    b2.info(f"**Storm kost tijd, vorst niet.** Op dagen met windstoten vanaf 60 km/u is {nl(storm)}% vertraagd, "
            f"op windstille dagen {nl(kalm)}% (2019). → pagina 5")
    b1.info(f"**Corona als experiment.** Van april t/m december 2020 bleef {nl(na.loc[2020, 'n'] / na.loc[2019, 'n'] * 100)}% "
            f"van de vluchten over en zakte het aandeel vertraagd van {nl(na.loc[2019, 'aandeel'] * 100)}% "
            f"naar {nl(na.loc[2020, 'aandeel'] * 100)}%. → pagina 6")
    b2.info("**Voorspellen lukt deels.** Het model herkent drukke, risicovolle momenten goed, maar de "
            "vertraging van één losse vlucht blijft grotendeels onvoorspelbaar. → pagina 7")


# ==========================================================================
# 2 · DATA-INSPECTIE
# ==========================================================================
def pagina_inspectie():
    st.title("Data-inspectie en opschonen")
    st.markdown("Eerst kijken wat er in de drie bronnen zit, dan pas conclusies trekken.")
    selectie_regel(filters_gelden=False)
    tab_vlucht, tab_haven, tab_weer, tab_klaar = st.tabs(
        ["Vluchten", "Luchthavens", "Weer", "Resultaat na opschonen"])
    koppeling = rapport["koppeling"]

    # ---------------- Vluchten ----------------
    with tab_vlucht:
        a, b, c = st.columns(3)
        a.metric("Rijen", nl(rapport["rijen_ruw"]))
        b.metric("Kolommen", rapport["kolommen_ruw"])
        c.metric("Periode", "2019 – 2020", "1 jan 2019 t/m 31 dec 2020", delta_color="off")
        st.markdown("##### Zo komt het bestand binnen")
        tabel(rapport["voorbeeld_ruw"])
        st.markdown("##### Per kolom")
        st.caption("Er staat geen enkele lege cel in het bestand. 'Leeg' is hier een streepje of `#N/A`.")
        tabel(rapport["inspectie_ruw"], height=35 * (len(rapport["inspectie_ruw"]) + 1) + 3)

        st.markdown("##### Wat viel op, en wat hebben we gedaan")
        stappen = pd.DataFrame([
            ["Streepje '-' betekent 'geen waarde' (gate, vertragingscode, baanconcept)",
             "Omgezet naar een echte lege waarde", nl(int(rapport["inspectie_ruw"]["Aantal '-'"].sum())) + " cellen"],
            ["'#N/A' als herkomst of bestemming", "Omgezet naar leeg; deze vluchten komen niet op de kaart",
             nl(koppeling["vluchten_zonder_code"]) + " vluchten"],
            ["Exact dubbele rijen", "Gecontroleerd, geen gevonden", nl(rapport["dubbele_rijen"])],
            ["Zelfde datum, tijd en vluchtnummer, andere werkelijke tijd",
             "Laten staan: niet te zien welke klopt", nl(rapport["dubbele_identifier"]) + " paren"],
            ["Datum en tijd staan als tekst in drie losse kolommen", "Samengevoegd tot een geplande en werkelijke datum-tijd",
             nl(rapport["rijen_ruw"]) + " rijen"],
            ["Werkelijke tijd na middernacht, maar er is alleen een geplande datum",
             "Een dag opgeteld bij de werkelijke tijd", nl(rapport["over_middernacht"]) + " vluchten"],
            ["Geen kolom voor vertraging", "Zelf gemaakt: werkelijk − gepland, in minuten", "nieuwe kolom"],
        ], columns=["Wat viel op", "Wat we deden", "Hoe vaak"])
        tabel(stappen)

        st.markdown("##### Het middernachtprobleem")
        m1, m2 = st.columns([1, 2])
        m1.metric("Kleinste 'vertraging' zonder correctie", nl(rapport["minimum_voor_correctie"]) + " min")
        m1.metric("Kleinste vertraging na correctie", nl(rapport["minimum_na_correctie"]) + " min")
        nacht = df[df["over_middernacht"]][["FLT", "gepland", "werkelijk", "vertraging_min"]].copy()
        nacht["zonder correctie (min)"] = nacht["vertraging_min"] - 24 * 60
        nacht = nacht.rename(columns={"FLT": "Vlucht", "gepland": "Gepland", "werkelijk": "Werkelijk",
                                      "vertraging_min": "Vertraging (min)"})
        with m2:
            st.caption("Een vlucht gepland om 22:45 die om 00:05 vertrekt is 80 minuten te laat, niet bijna 23 uur te vroeg.")
            nacht[["Vertraging (min)", "zonder correctie (min)"]] = nacht[["Vertraging (min)", "zonder correctie (min)"]].round(1)
            tabel(nacht)

        st.markdown("##### De nieuwe kolom: vertraging in minuten")
        # Zelf tellen per 5 minuten: dat is sneller dan 323.000 losse punten naar de browser sturen
        randen = np.arange(-60, 125, 5)
        aantallen, _ = np.histogram(df["vertraging_min"].clip(-60, 119.9), bins=randen)
        fig = go.Figure(go.Bar(x=randen[:-1] + 2.5, y=aantallen, width=5,
                               marker=dict(color=BLAUW, line=dict(color="white", width=1)),
                               customdata=np.stack([randen[:-1], randen[1:]], axis=-1),
                               hovertemplate="%{customdata[0]} tot %{customdata[1]} min<br>%{y:,} vluchten<extra></extra>"))
        opmaak(fig, x_titel="Vertraging in minuten (negatief = te vroeg; alles boven 120 staat in de laatste staaf)",
               y_titel="Aantal vluchten", hoogte=320, legenda=False)
        for x, tekst in [(0, "op tijd"), (VERTRAAGD_VANAF_MIN, "vanaf hier 'vertraagd'")]:
            fig.add_shape(type="line", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(color=INKT, width=1, dash="dot"))
            fig.add_annotation(x=x, yref="paper", y=1, text=tekst, showarrow=False, xanchor="left", xshift=4,
                               yanchor="top", font=dict(size=12, color=INKT))
        toon(fig)
        st.caption(
            f"De helft van de vluchten zit tussen {nl(df['vertraging_min'].quantile(0.25))} en "
            f"{nl(df['vertraging_min'].quantile(0.75))} minuten. De staart naar rechts is lang: 1% heeft meer dan "
            f"{nl(df['vertraging_min'].quantile(0.99))} minuten vertraging, de grootste is {nl(rapport['maximum_na_correctie'] / 60, 1)} uur. "
            "Daarom rekenen we vooral met het aandeel vertraagde vluchten en de mediaan, niet alleen met het gemiddelde."
        )

    # ---------------- Luchthavens ----------------
    with tab_haven:
        a, b, c = st.columns(3)
        a.metric("Locaties in het bestand", nl(rapport["luchthavens_rijen"]))
        b.metric("Bestemmingen in de vluchtdata", koppeling["bestemmingen"])
        c.metric("Vluchten met een luchthaven", nl(100 * (koppeling["vluchten_via_icao"] + koppeling["vluchten_via_iata"]) / rapport["rijen_ruw"], 2) + "%")
        st.markdown(
            "Het bestand heeft **geen kopregel** en gebruikt `\\N` voor een ontbrekende waarde. De kolomnamen hebben we zelf "
            "toegevoegd. Het bevat ook niet alleen vliegvelden:"
        )
        soorten = pd.DataFrame({"Soort": list(rapport["luchthavens_type"]), "Aantal": list(rapport["luchthavens_type"].values())})
        soorten["Soort"] = soorten["Soort"].map({"airport": "Vliegveld", "station": "Treinstation",
                                                 "port": "Haven", "unknown": "Onbekend"}).fillna(soorten["Soort"])
        tabel(soorten)

        st.markdown("##### Koppelen: ICAO of IATA?")
        st.markdown(
            f"`Org/Des` bevat bijna altijd een code van vier letters. Koppelen op de **ICAO**-kolom vindt "
            f"**{koppeling['bestemmingen_via_icao']}** van de {koppeling['bestemmingen']} bestemmingen; koppelen op de "
            f"**IATA**-kolom maar **{koppeling['bestemmingen_via_iata_kolom']}**. Die paar zijn precies de codes van drie letters "
            "(zoals `BER`). We koppelen dus eerst op ICAO en vullen de rest aan via IATA."
        )
        resultaat = pd.DataFrame([
            ["Gekoppeld via ICAO", koppeling["vluchten_via_icao"]],
            ["Aangevuld via IATA", koppeling["vluchten_via_iata"]],
            ["Geen code in de bron (#N/A)", koppeling["vluchten_zonder_code"]],
            ["Code staat niet in het luchthavenbestand", koppeling["vluchten_code_onbekend"]],
        ], columns=["Uitkomst", "Vluchten"])
        tabel(resultaat)
        onbekend = ", ".join(f"{code} ({n}×)" for code, n in koppeling["onbekende_codes"].items())
        st.caption(f"Niet gevonden: {onbekend}. Dat zijn {koppeling['vluchten_code_onbekend']} vluchten; die laten we buiten de kaart.")

    # ---------------- Weer ----------------
    with tab_weer:
        weer_info = rapport["weer"]
        ruw_weer = laad_ruw_weer()
        a, b, c = st.columns(3)
        a.metric("Dagen", weer_info["dagen"])
        b.metric("Dagen die ontbreken", 731 - weer_info["dagen"])
        c.metric("Vluchten met weer erbij", nl(100 * rapport["weer_dagen_gekoppeld"] / rapport["rijen_schoon"], 1) + "%")
        st.markdown(
            "Dagwaarden van Meteostat, station 06670 (Zürich-Kloten), voor precies de periode van de vluchten: "
            "1 januari 2019 t/m 31 december 2020. Elke dag is aanwezig."
        )
        leeg = pd.DataFrame({"Kolom": list(weer_info["leeg_per_kolom"]), "Lege dagen": list(weer_info["leeg_per_kolom"].values())})
        leeg["Leeg (%)"] = (100 * leeg["Lege dagen"] / weer_info["dagen"]).round(1)
        leeg["Wat we deden"] = leeg["Kolom"].map({"snow": "Kolom weggelaten", "tsun": "Kolom weggelaten",
                                                  "prcp": "Leeg gelaten"}).fillna("")
        tabel(leeg)

        st.markdown("##### Neerslag: vanaf 11 november 2020 klopt er iets niet")
        fig = go.Figure(go.Scatter(x=ruw_weer["date"], y=ruw_weer["prcp"], mode="lines", line=dict(color=BLAUW, width=1.5),
                                   name="Neerslag", hovertemplate="%{y:,.1f} mm<extra></extra>"))
        fig.add_shape(type="rect", x0="2020-11-11", x1="2020-12-31", yref="paper", y0=0, y1=1,
                      fillcolor=ORANJE, opacity=0.15, line=dict(width=0))
        fig.add_annotation(x="2020-11-11", yref="paper", y=1, text="51 dagen op rij ± 21,6 mm", showarrow=False,
                           xanchor="right", xshift=-6, yanchor="top", font=dict(size=12, color=INKT))
        opmaak(fig, y_titel="Neerslag per dag (mm)", hoogte=320, legenda=False)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(hoverformat="%d-%m-%Y", dtick="M3", tickformat="%m-%Y")
        toon(fig)
        st.caption(
            f"In de oranje periode valt er volgens de bron gemiddeld {nl(weer_info['neerslag_verdacht_gemiddeld'], 1)} mm per dag, "
            f"tegen {nl(weer_info['neerslag_normaal_gemiddeld'], 1)} mm in de rest van de twee jaar. Dat is geen weer maar een "
            f"opvulwaarde. Voor die {weer_info['neerslag_verdacht_dagen']} dagen hebben we de neerslag op 'onbekend' gezet; "
            "temperatuur, wind en luchtdruk zien er daar wel gewoon uit."
        )

    # ---------------- Resultaat ----------------
    with tab_klaar:
        a, b, c = st.columns(3)
        a.metric("Rijen", nl(rapport["rijen_schoon"]), "geen rij weggegooid", delta_color="off")
        b.metric("Kolommen", rapport["kolommen_schoon"])
        c.metric("Bronnen gecombineerd", 3)
        st.markdown("##### Wat nog leeg is, en waarom")
        leeg = rapport["leeg_na_opschonen"]
        leeg = leeg[leeg > 0].rename("Lege waarden").reset_index().rename(columns={"index": "Kolom"})
        uitleg = {
            "DL1": "Vertragingscode: alleen ingevuld als er een oorzaak is geregistreerd",
            "IX1": "Idem", "DL2": "Idem (tweede oorzaak)", "IX2": "Idem",
            "TAR": "Geen geplande gate bekend", "GAT": "Geen werkelijke gate bekend", "RWC": "Geen baanconcept bekend",
            "Org/Des": "#N/A in de bron", "prcp": "Opvulwaarde in de bron (zie tabblad Weer)",
            "IATA": "Luchthaven heeft geen IATA-code", "ICAO": "Luchthaven gekoppeld via IATA",
        }
        leeg["Reden"] = leeg["Kolom"].map(uitleg).fillna("Vlucht zonder gekoppelde luchthaven")
        leeg["Leeg (%)"] = (100 * leeg["Lege waarden"] / rapport["rijen_schoon"]).round(2)
        tabel(leeg)
        st.markdown("##### De gecombineerde dataset")
        tabel(df.head(50))


# ==========================================================================
# 3 · TIJD EN DRUKTE
# ==========================================================================
def pagina_tijd():
    st.title("Wanneer ontstaat vertraging?")
    selectie_regel()
    stop_als_leeg(selectie)

    # --- Per uur van de dag -------------------------------------------------
    st.markdown("#### Door de dag heen")
    maat = st.radio("Toon", list(MATEN)[:2], horizontal=True, key="tijd_maat")
    kolom, eenheid, decimalen = MATEN[maat]
    per_uur = samenvatting(selectie, ["richting", "uur"])
    per_uur = per_uur[per_uur["vluchten"] >= 100]
    stop_als_leeg(per_uur, minimum=2)

    uren_bereik = [per_uur["uur"].min() - 0.7, per_uur["uur"].max() + 0.7]
    fig = go.Figure()
    einden = []
    for richting, deel in per_uur.groupby("richting", observed=True):
        kleur = KLEUR_RICHTING[str(richting)]
        fig.add_trace(go.Scatter(
            x=deel["uur"], y=deel[kolom], mode="lines+markers", name=str(richting),
            line=dict(color=kleur, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
            customdata=deel["vluchten"], hovertemplate="%{y:,.1f}" + eenheid + " (%{customdata:,} vluchten)",
        ))
        einden.append((deel["uur"].iloc[-1], deel[kolom].iloc[-1], str(richting), kleur))
    eindlabels(fig, einden)
    opmaak(fig, x_titel="Gepland uur", y_titel=maat, hoogte=360, marge_rechts=80)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1, ticksuffix=":00", range=uren_bereik)
    toon(fig)

    fig = go.Figure()
    for richting, deel in per_uur.groupby("richting", observed=True):
        fig.add_trace(go.Bar(x=deel["uur"], y=deel["vluchten"], name=str(richting),
                             marker=dict(color=KLEUR_RICHTING[str(richting)], line=dict(color="white", width=1)),
                             hovertemplate="%{y:,} vluchten"))
    opmaak(fig, x_titel="Gepland uur", y_titel="Aantal vluchten", hoogte=230, marge_rechts=80)
    fig.update_layout(barmode="group", hovermode="x unified", bargap=0.25)
    fig.update_xaxes(dtick=1, ticksuffix=":00", range=uren_bereik)
    toon(fig)

    piek = per_uur.loc[per_uur[kolom].idxmax()]
    dal = per_uur.loc[per_uur[kolom].idxmin()]
    st.caption(
        f"Hoogste waarde: **{str(piek['richting']).lower()} om {int(piek['uur'])}:00** ({nl(piek[kolom], 1)}{eenheid}); "
        f"laagste: {str(dal['richting']).lower()} om {int(dal['uur'])}:00 ({nl(dal[kolom], 1)}{eenheid}). "
        "De staven laten zien dat Zürich in golven werkt: eerst een golf aankomsten, kort daarna een golf vertrekken."
    )

    # --- Drukte tegen vertraging --------------------------------------------
    st.markdown("#### Hoe drukker het uur, hoe meer vertraging")
    data = selectie.copy()
    data["drukteklasse"] = pd.cut(data["drukte"], [0, 20, 30, 40, 50, 60, 100], right=True,
                                  labels=["1–20", "21–30", "31–40", "41–50", "51–60", "meer dan 60"])
    per_drukte = samenvatting(data, ["richting", "drukteklasse"])
    per_drukte = per_drukte[per_drukte["vluchten"] >= 200]
    fig = go.Figure()
    for richting, deel in per_drukte.groupby("richting", observed=True):
        kleur = KLEUR_RICHTING[str(richting)]
        fig.add_trace(go.Scatter(
            x=deel["drukteklasse"].astype(str), y=deel["aandeel"], mode="lines+markers", name=str(richting),
            line=dict(color=kleur, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
            customdata=deel["vluchten"], hovertemplate="%{y:,.1f}% vertraagd (%{customdata:,} vluchten)",
        ))
    opmaak(fig, x_titel="Aantal geplande vluchten in hetzelfde uur (start + landing)",
           y_titel="Aandeel vertraagd (%)", hoogte=340)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(categoryorder="array", categoryarray=["1–20", "21–30", "31–40", "41–50", "51–60", "meer dan 60"])
    toon(fig)
    totaal = samenvatting(data, "drukteklasse")
    totaal = totaal[totaal["vluchten"] >= 200]
    tekst = "Deze kolom zit niet in de bron: we hebben per uur geteld hoeveel vluchten er gepland stonden. "
    if len(totaal) >= 2:
        rustigst, drukst = totaal.iloc[0], totaal.iloc[-1]
        tekst += (f"In de rustigste klasse ({rustigst['drukteklasse']} vluchten per uur) is {nl(rustigst['aandeel'])}% vertraagd, "
                  f"in de drukste ({drukst['drukteklasse']}) {nl(drukst['aandeel'])}%. ")
    st.caption(tekst + "Klassen met minder dan 200 vluchten zijn weggelaten.")

    # --- Dag van de week x uur ----------------------------------------------
    st.markdown("#### Welke dag, welk uur?")
    per_cel = samenvatting(selectie, ["weekdag", "uur"])
    per_cel = per_cel[(per_cel["vluchten"] >= 30) & per_cel["uur"].between(6, 22)]
    raster = per_cel.pivot(index="weekdag", columns="uur", values="aandeel").reindex(index=range(7), columns=range(6, 23))
    aantallen = per_cel.pivot(index="weekdag", columns="uur", values="vluchten").reindex(index=range(7), columns=range(6, 23))
    fig = go.Figure(go.Heatmap(
        z=raster.to_numpy(), x=[f"{u}:00" for u in raster.columns], y=WEEKDAGEN, customdata=aantallen.to_numpy(),
        colorscale=SCHAAL_BLAUW, xgap=2, ygap=2, colorbar=dict(title="% vertraagd", thickness=12),
        hovertemplate="%{y} %{x}<br>%{z:,.1f}% vertraagd<br>%{customdata:,} vluchten<extra></extra>",
    ))
    opmaak(fig, hoogte=320, legenda=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    toon(fig)
    per_dag = samenvatting(selectie, "weekdag")
    best, slechtst = per_dag.loc[per_dag["aandeel"].idxmin()], per_dag.loc[per_dag["aandeel"].idxmax()]
    st.caption(
        f"Rustigste dag: {WEEKDAGEN[int(best['weekdag'])].lower()} ({nl(best['aandeel'], 1)}% vertraagd); "
        f"slechtste dag: {WEEKDAGEN[int(slechtst['weekdag'])].lower()} ({nl(slechtst['aandeel'], 1)}%). "
        "Vakjes met minder dan 30 vluchten blijven leeg."
    )

    # --- Vergelijken op toestel, maatschappij of baan -------------------------
    st.markdown("#### Verschillen tussen toestellen, maatschappijen en banen")
    c1, c2 = st.columns([1, 2])
    soort = c1.selectbox("Vergelijk op", ["Baan", "Vliegtuigtype", "Maatschappij"])
    aantal = c2.slider("Toon de grootste", 5, 20, 12, disabled=soort == "Baan")
    kolomnaam = {"Baan": "RWY", "Vliegtuigtype": "ACT", "Maatschappij": "maatschappij"}[soort]
    per_groep = samenvatting(selectie, kolomnaam)
    per_groep = per_groep[per_groep["vluchten"] >= 100].nlargest(aantal, "vluchten").sort_values("aandeel")
    fig = go.Figure(go.Bar(
        x=per_groep["aandeel"], y=per_groep[kolomnaam].astype(str), orientation="h",
        marker=dict(color=BLAUW), text=[nl(w, 1) + "%" for w in per_groep["aandeel"]], textposition="outside",
        cliponaxis=False, customdata=np.stack([per_groep["vluchten"], per_groep["gem"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>%{x:,.1f}% vertraagd<br>%{customdata[0]:,} vluchten<br>"
                      "gemiddeld %{customdata[1]:,.1f} min<extra></extra>",
    ))
    opmaak(fig, x_titel="Aandeel vertraagd (%)", hoogte=max(260, 30 * len(per_groep) + 80), legenda=False, marge_rechts=50)
    fig.update_yaxes(type="category", showgrid=False)
    toon(fig)
    st.caption(
        "Let op: dit zijn samenhangen, geen oorzaken. Grote toestellen vliegen ver en vertrekken in de drukke middaggolf; "
        "baan 10 wordt vooral gebruikt als de wind ongunstig staat. Het model op pagina 7 weegt die factoren tegen elkaar af."
    )


# ==========================================================================
# 4 · BESTEMMINGEN (KAART)
# ==========================================================================
def pagina_kaart():
    st.title("Waarheen, en waar loopt het uit?")
    selectie_regel()
    stop_als_leeg(selectie)

    c1, c2, c3 = st.columns([1.2, 1.5, 1.3])
    gebied = c1.radio("Gebied", ["Europa en omgeving", "Intercontinentaal"],
                      help=f"Grens: {nl(LANGE_AFSTAND_KM)} km vanaf Zürich.")
    kleur_keuze = c2.radio("Kleur toont", ["Aandeel vertraagd (%)", "Gemiddelde vertraging (min)",
                                           "Verandering aantal vluchten 2020 t.o.v. 2019 (%)"])
    minimum = c3.slider("Minimaal aantal vluchten per bestemming", 10, 1000, 200 if gebied == "Europa en omgeving" else 50, step=10,
                        help="Bestemmingen met weinig vluchten geven toevallige uitschieters.")

    st.caption(
        "**Waarom twee kaarten?** 295 bestemmingen op één wereldkaart wordt een wolk stippen. "
        f"In Europa en omgeving zit {nl((df['gebied'] == 'Europa en omgeving').sum() / df['gebied'].notna().sum() * 100)}% van de vluchten "
        "en zie je de dichtheid; intercontinentaal zijn het er weinig, maar elke lijn is een verbinding."
    )

    info = df.drop_duplicates("Org/Des")[["Org/Des", "Name", "City", "Country", "Latitude", "Longitude", "afstand_km", "gebied"]].copy()
    info["Org/Des"] = info["Org/Des"].astype(str)

    verandering = kleur_keuze.startswith("Verandering")
    if verandering:
        # Voor de vergelijking tussen de jaren negeren we het jaarfilter
        per_jaar = zonder_jaar.groupby(["Org/Des", "jaar"], observed=True).size().unstack("jaar", fill_value=0)
        per_jaar = per_jaar.reindex(columns=[2019, 2020], fill_value=0)
        stats = samenvatting(zonder_jaar, "Org/Des")
        stats["Org/Des"] = stats["Org/Des"].astype(str)
        per_jaar.index = per_jaar.index.astype(str)
        stats["v2019"] = stats["Org/Des"].map(per_jaar[2019])
        stats["v2020"] = stats["Org/Des"].map(per_jaar[2020])
        stats = stats[stats["v2019"] >= minimum]
        stats["waarde"] = (stats["v2020"] - stats["v2019"]) / stats["v2019"] * 100
        stats["grootte"] = stats["v2019"]
        schaal, eenheid, legenda_titel = SCHAAL_ROOD_BLAUW, "%", "Verandering (%)"
        kleur_min, kleur_max = -100, 100
    else:
        stats = samenvatting(selectie, "Org/Des")
        stats["Org/Des"] = stats["Org/Des"].astype(str)
        stats = stats[stats["vluchten"] >= minimum]
        stats["waarde"] = stats["aandeel"] if kleur_keuze.startswith("Aandeel") else stats["gem"]
        stats["grootte"] = stats["vluchten"]
        schaal = SCHAAL_BLAUW
        eenheid, legenda_titel = ("%", "% vertraagd") if kleur_keuze.startswith("Aandeel") else (" min", "Gem. vertraging (min)")
        kleur_min, kleur_max = None, None

    best = stats.merge(info, on="Org/Des", how="inner")
    best = best[(best["gebied"] == gebied) & best["Latitude"].notna()].sort_values("grootte", ascending=False)
    if len(best) == 0:
        st.warning("Geen bestemmingen bij deze instellingen. Zet het minimum aantal vluchten lager.")
        st.stop()
    if kleur_min is None:
        kleur_min, kleur_max = float(np.floor(best["waarde"].quantile(0.05))), float(np.ceil(best["waarde"].quantile(0.95)))
        if kleur_min == kleur_max:
            kleur_max = kleur_min + 1

    zurich_lat, zurich_lon = rapport["koppeling"]["zurich"]
    fig = go.Figure()
    if gebied == "Intercontinentaal":
        # Een lijn per verbinding; None scheidt de lijnen binnen een trace
        lons, lats = [], []
        for _, rij in best.iterrows():
            lons += [zurich_lon, rij["Longitude"], None]
            lats += [zurich_lat, rij["Latitude"], None]
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", line=dict(width=0.7, color="rgba(82,81,78,0.35)"),
                                    hoverinfo="skip", showlegend=False))
    grootste_stip = 34 if gebied == "Europa en omgeving" else 26
    fig.add_trace(go.Scattergeo(
        lon=best["Longitude"], lat=best["Latitude"], mode="markers", showlegend=False,
        marker=dict(
            size=best["grootte"], sizemode="area", sizeref=2.0 * best["grootte"].max() / grootste_stip ** 2, sizemin=4,
            color=best["waarde"], colorscale=schaal, cmin=kleur_min, cmax=kleur_max,
            colorbar=dict(title=legenda_titel, thickness=12, len=0.6), line=dict(color=BLAUW_DONKER, width=0.6), opacity=0.92,
        ),
        customdata=hoverdata(best, ["City", "Name", "Country", "vluchten", "aandeel", "gem", "afstand_km", "waarde"]),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}, %{customdata[6]:,.0f} km<br>"
                      "%{customdata[3]:,} vluchten<br>%{customdata[4]:,.1f}% vertraagd · gemiddeld %{customdata[5]:,.1f} min"
                      + ("<br>Verandering 2020: %{customdata[7]:+,.0f}%" if verandering else "") + "<extra></extra>",
    ))
    # Alleen de grootste bestemmingen krijgen een naam, en alleen als er nog geen naam vlakbij staat.
    # Anders staan de namen in het drukke midden van Europa over elkaar heen.
    ruimte = 5.0 if gebied == "Europa en omgeving" else 14.0       # in graden
    met_naam = []
    for _, rij in best.iterrows():                                  # best is gesorteerd op grootte
        bij_zurich = abs(rij["Latitude"] - zurich_lat) < ruimte * 0.5 and abs(rij["Longitude"] - zurich_lon) < ruimte * 0.8
        vrij = all(abs(rij["Latitude"] - ander["Latitude"]) > ruimte * 0.5 or abs(rij["Longitude"] - ander["Longitude"]) > ruimte
                   for ander in met_naam)
        if vrij and not bij_zurich:
            met_naam.append(rij)
        if len(met_naam) == 14:
            break
    top = pd.DataFrame(met_naam) if met_naam else best.head(0)
    fig.add_trace(go.Scattergeo(lon=top["Longitude"], lat=top["Latitude"], mode="text", text=top["City"].astype(str),
                                textposition="top center", textfont=dict(size=11, color=INKT), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lon=[zurich_lon], lat=[zurich_lat], mode="markers+text", text=["Zürich"],
                                textposition="bottom center", textfont=dict(size=12, color=INKT),
                                marker=dict(size=11, color=INKT, symbol="diamond", line=dict(color="white", width=1.5)),
                                hovertemplate="<b>Zürich Airport</b><extra></extra>", showlegend=False))
    opmaak(fig, hoogte=600 if gebied == "Europa en omgeving" else 470, legenda=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_geos(showland=True, landcolor=LAND, showcountries=True, countrycolor="white", coastlinecolor="#c9c7c0",
                    showocean=True, oceancolor="#f8fafc", showframe=False, resolution=50)
    if gebied == "Europa en omgeving":
        fig.update_geos(projection_type="mercator", lonaxis_range=[-32, 58], lataxis_range=[26, 67])
    else:
        fig.update_geos(projection_type="natural earth", lataxis_range=[-45, 75])
    toon(fig)

    uitleg_grootte = "vluchten in 2019" if verandering else "vluchten in de selectie"
    st.caption(f"Elke stip is een bestemming; hoe groter de stip, hoe meer {uitleg_grootte}. "
               f"Getoond: {len(best)} bestemmingen met minstens {minimum} vluchten. Zweef over een stip voor de cijfers.")

    # --- Onder de kaart: ranglijst en detail van een bestemming --------------
    links, rechts = st.columns([1, 1.2])
    with links:
        st.markdown("##### Uitschieters op de kaart")
        volgorde = st.radio("Sorteer", ["Hoogste eerst", "Laagste eerst"], horizontal=True, label_visibility="collapsed")
        lijst = best.sort_values("waarde", ascending=volgorde == "Laagste eerst").head(10)
        kolommen = ["City", "Country", "vluchten", "aandeel", "gem"] + (["v2020", "waarde"] if verandering else [])
        lijst = lijst[kolommen].rename(columns={
            "City": "Stad", "Country": "Land", "vluchten": "Vluchten", "aandeel": "% vertraagd", "gem": "Gem. min",
            "v2020": "Vluchten 2020", "waarde": "Verandering (%)"})
        if verandering:
            lijst = lijst.rename(columns={"Vluchten": "Vluchten 2019 + 2020"})
        tabel(lijst.round(1))
        st.caption(f"Gesorteerd op: {kleur_keuze.lower()}.")
    with rechts:
        st.markdown("##### Eén bestemming door de tijd")
        opties = best["Org/Des"].tolist()
        labels = dict(zip(best["Org/Des"], best["City"].astype(str) + " · " + best["Name"].astype(str)))
        code = st.selectbox("Bestemming", opties, format_func=lambda c: labels[c], label_visibility="collapsed")
        route = zonder_jaar[zonder_jaar["Org/Des"] == code]
        per_maand = route.groupby(pd.Grouper(key="datum", freq="MS")).agg(
            vluchten=("vertraagd", "size"), aandeel=("vertraagd", "mean"))
        per_maand = per_maand.reindex(pd.date_range("2019-01-01", "2020-12-01", freq="MS"), fill_value=0)
        fig = go.Figure(go.Scatter(x=per_maand.index, y=per_maand["vluchten"], mode="lines+markers",
                                   line=dict(color=BLAUW, width=2), marker=dict(size=6),
                                   hovertemplate="%{y:,} vluchten<extra></extra>"))
        opmaak(fig, y_titel="Vluchten per maand", hoogte=260, legenda=False)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(hoverformat="%m-%Y", dtick="M6", tickformat="%m-%Y")
        lockdown_lijn(fig, "lockdown")
        toon(fig)
        rij = best[best["Org/Des"] == code].iloc[0]
        st.caption(f"{labels[code]}: {nl(rij['afstand_km'])} km, {nl(rij['vluchten'])} vluchten, "
                   f"{nl(rij['aandeel'], 1)}% vertraagd, gemiddeld {nl(rij['gem'], 1)} minuten.")


# ==========================================================================
# 5 · WEER
# ==========================================================================
WEER_KLASSEN = {
    "wpgt": ([0, 30, 40, 50, 60, 75, 250], ["< 30", "30–40", "40–50", "50–60", "60–75", "≥ 75"]),
    "prcp": ([0, 1, 5, 15, 250], ["droog (< 1)", "1–5", "5–15", "≥ 15"]),
    "tavg": ([-20, 0, 5, 10, 15, 20, 40], ["< 0", "0–5", "5–10", "10–15", "15–20", "≥ 20"]),
    "tmin": ([-20, -3, 0, 5, 10, 15, 40], ["< −3", "−3–0", "0–5", "5–10", "10–15", "≥ 15"]),
    "wspd": ([0, 5, 10, 15, 20, 60], ["< 5", "5–10", "10–15", "15–20", "≥ 20"]),
    "pres": ([900, 1005, 1012, 1018, 1025, 1100], ["< 1005", "1005–1012", "1012–1018", "1018–1025", "≥ 1025"]),
}


def pagina_weer():
    st.title("Wat doet het weer?")
    selectie_regel()
    stop_als_leeg(selectie)
    st.markdown(
        "Elke vlucht heeft het weer van zijn dag in Zürich gekregen. Het weer staat in geen van de vluchtkolommen, "
        "en het weerbestand weet niets van vluchten: dit verband bestaat alleen in de combinatie."
    )

    c1, c2 = st.columns([1.5, 1])
    variabele = c1.selectbox("Weervariabele", ["wpgt", "prcp", "tavg", "tmin", "wspd", "pres"],
                             format_func=lambda k: WEER_KOLOMMEN[k])
    splitsen = c2.checkbox("Splits in rustige en drukke dagen", value=True,
                           help="Druk = meer vluchten op die dag dan de middelste dag in de selectie.")
    randen, labels = WEER_KLASSEN[variabele]
    data = selectie.dropna(subset=[variabele]).copy()
    data["klasse"] = pd.cut(data[variabele], randen, right=False, labels=labels)
    data["vluchten_dag"] = data.groupby("datum")["vertraagd"].transform("size")
    grens = data.drop_duplicates("datum")["vluchten_dag"].median()
    data["soort_dag"] = np.where(data["vluchten_dag"] > grens, "Drukke dag", "Rustige dag")

    groep = ["soort_dag", "klasse"] if splitsen else ["klasse"]
    per_klasse = data.groupby(groep, observed=True).agg(
        vluchten=("vertraagd", "size"), dagen=("datum", "nunique"), aandeel=("vertraagd", "mean")).reset_index()
    per_klasse["aandeel"] *= 100
    per_klasse = per_klasse[per_klasse["dagen"] >= 3]

    st.markdown(f"#### Aandeel vertraagd naar {WEER_KOLOMMEN[variabele].lower()}")
    fig = go.Figure()
    reeksen = [("Rustige dag", BLAUW_LICHT), ("Drukke dag", BLAUW_DONKER)] if splitsen else [("Alle dagen", BLAUW)]
    for naam, kleur in reeksen:
        deel = per_klasse[per_klasse["soort_dag"] == naam] if splitsen else per_klasse
        fig.add_trace(go.Bar(
            x=deel["klasse"].astype(str), y=deel["aandeel"], name=naam,
            marker=dict(color=kleur, line=dict(color="white", width=1)),
            text=[nl(w) + "%" for w in deel["aandeel"]], textposition="outside", cliponaxis=False,
            customdata=np.stack([deel["dagen"], deel["vluchten"]], axis=-1),
            hovertemplate="<b>" + naam + "</b>, %{x}<br>%{y:,.1f}% vertraagd<br>%{customdata[0]} dagen, %{customdata[1]:,} vluchten<extra></extra>",
        ))
    opmaak(fig, x_titel=WEER_KOLOMMEN[variabele], y_titel="Aandeel vertraagd (%)", hoogte=380, legenda=splitsen)
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    toon(fig)
    if splitsen:
        st.caption(f"Druk = meer dan {nl(grens)} vluchten op die dag. Klassen met minder dan 3 dagen zijn weggelaten; "
                   "zweef over een staaf voor het aantal dagen erachter.")

    # Vaste bevinding over wind en drukte, op 2019 (normaal jaar)
    d19 = df[df["jaar"] == 2019].copy()
    d19["vluchten_dag"] = d19.groupby("datum")["vertraagd"].transform("size")
    grens19 = d19.drop_duplicates("datum")["vluchten_dag"].median()
    druk_storm = d19[(d19["vluchten_dag"] > grens19) & (d19["wpgt"] >= 60)]["vertraagd"].mean() * 100
    rustig_kalm = d19[(d19["vluchten_dag"] <= grens19) & (d19["wpgt"] < 40)]["vertraagd"].mean() * 100
    vorst, geen_vorst = d19[d19["vorst"]]["vertraagd"].mean() * 100, d19[~d19["vorst"]]["vertraagd"].mean() * 100
    a, b = st.columns(2)
    a.info(f"**Wind en drukte versterken elkaar.** Op een drukke dag met windstoten vanaf 60 km/u is {nl(druk_storm)}% "
           f"vertraagd; op een rustige, windstille dag {nl(rustig_kalm)}% (2019).")
    b.info(f"**Kou is niet het probleem.** Op vorstdagen is {nl(vorst)}% vertraagd, op andere dagen {nl(geen_vorst)}% (2019). "
           "In de winter wordt minder gevlogen; de zomer is drukker én heeft buien. Kijk daarom altijd naar weer en drukte samen.")

    # --- Elke dag een stip ---------------------------------------------------
    st.markdown("#### Elke dag een stip: drukte, weer en vertraging in één beeld")
    per_dag = selectie.groupby("datum").agg(
        vluchten=("vertraagd", "size"), aandeel=("vertraagd", "mean"), wpgt=("wpgt", "first"), prcp=("prcp", "first"),
        tavg=("tavg", "first")).reset_index()
    per_dag = per_dag[per_dag["vluchten"] >= 50].copy()
    if len(per_dag) == 0:
        st.info("In deze selectie is er geen enkele dag met 50 of meer vluchten. Maak de selectie ruimer voor de dagvergelijking.")
        st.stop()
    per_dag["aandeel"] *= 100
    per_dag["dag"] = per_dag["datum"].dt.strftime("%d-%m-%Y")
    per_dag["windklasse"] = pd.cut(per_dag["wpgt"], [0, 40, 60, 250], right=False,
                                   labels=["Windstoten < 40 km/u", "40–60 km/u", "≥ 60 km/u"])
    fig = go.Figure()
    for klasse, kleur in zip(["Windstoten < 40 km/u", "40–60 km/u", "≥ 60 km/u"], [BLAUW_LICHT, BLAUW_MIDDEN, BLAUW_DONKER]):
        deel = per_dag[per_dag["windklasse"] == klasse]
        if len(deel) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=deel["vluchten"], y=deel["aandeel"], mode="markers", name=klasse,
            marker=dict(size=8, color=kleur, line=dict(color="white", width=1)),
            customdata=hoverdata(deel, ["dag", "wpgt", "tavg"]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:,} vluchten, %{y:,.1f}% vertraagd<br>"
                          "windstoten %{customdata[1]} km/u · %{customdata[2]} °C<extra></extra>",
        ))
    opmaak(fig, x_titel="Vluchten op die dag", y_titel="Aandeel vertraagd (%)", hoogte=420)
    toon(fig)
    st.caption("Elke stip is een dag: naar rechts meer vluchten, naar boven meer vertraging, donkerder is meer wind. "
               "Dagen met minder dan 50 vluchten in de selectie zijn weggelaten.")

    st.markdown("#### De tien slechtste dagen")
    slechtst = per_dag.nlargest(10, "aandeel")[["datum", "vluchten", "aandeel", "wpgt", "prcp", "tavg"]].copy()
    slechtst["datum"] = slechtst["datum"].dt.strftime("%d-%m-%Y")
    slechtst.columns = ["Datum", "Vluchten", "% vertraagd", "Windstoten (km/u)", "Neerslag (mm)", "Temperatuur (°C)"]
    tabel(slechtst.round(1))
    st.caption("Niet elke slechte dag is aan het dagweer te zien. Een onweersbui van een uur of een storing verdwijnt in een daggemiddelde; "
               "dat is een grens van deze weerbron.")


# ==========================================================================
# 6 · 2019 TEGENOVER 2020
# ==========================================================================
def pagina_jaren():
    st.title("2019 tegenover 2020")
    st.caption("Deze pagina vergelijkt altijd beide jaren. De filters voor richting, maatschappij en vliegtuigtype gelden wel.")
    stop_als_leeg(zonder_jaar)
    if zonder_jaar["jaar"].nunique() < 2:
        st.warning("Met deze filters is er maar één jaar over.")
        st.stop()

    per_jaar = zonder_jaar.groupby("jaar").agg(
        vluchten=("vertraagd", "size"), aandeel=("vertraagd", "mean"), gem=("vertraging_min", "mean"),
        mediaan=("vertraging_min", "median"), bestemmingen=("Org/Des", "nunique"))
    k = st.columns(5)
    for kolom_st, (titel, veld, factor, eenheid, dec) in zip(k, [
        ("Vluchten", "vluchten", 1, "", 0), ("Vertraagd", "aandeel", 100, "%", 1), ("Gemiddelde vertraging", "gem", 1, " min", 1),
        ("Mediaan", "mediaan", 1, " min", 1), ("Bestemmingen", "bestemmingen", 1, "", 0)]):
        v19, v20 = per_jaar.loc[2019, veld] * factor, per_jaar.loc[2020, veld] * factor
        verschil_eenheid = " %-punt" if veld == "aandeel" else eenheid
        kolom_st.metric(f"{titel} 2020", nl(v20, dec) + eenheid, f"{nl(v20 - v19, dec)}{verschil_eenheid} t.o.v. 2019",
                        delta_color="off" if veld in ("vluchten", "bestemmingen") else "inverse")

    # --- Lijngrafiek per maand ------------------------------------------------
    st.markdown("#### Maand voor maand")
    keuzes = dict(MATEN)
    keuzes["Aantal bestemmingen"] = ("bestemmingen", "", 0)
    maat = st.radio("Toon", list(keuzes), horizontal=True, key="jaar_maat")
    kolom, eenheid, decimalen = keuzes[maat]
    per_maand = zonder_jaar.groupby(["jaar", "maand"]).agg(
        vluchten=("vertraagd", "size"), aandeel=("vertraagd", "mean"), gem=("vertraging_min", "mean"),
        bestemmingen=("Org/Des", "nunique")).reset_index()
    per_maand["aandeel"] *= 100

    fig = go.Figure()
    for jaar in (2019, 2020):
        deel = per_maand[per_maand["jaar"] == jaar]
        fig.add_trace(go.Scatter(
            x=[MAANDEN[m - 1] for m in deel["maand"]], y=deel[kolom], mode="lines+markers", name=str(jaar),
            line=dict(color=KLEUR_JAAR[jaar], width=2.5 if jaar == 2020 else 2),
            marker=dict(size=8, line=dict(color="white", width=2)),
            customdata=deel["vluchten"], hovertemplate="%{y:,." + str(decimalen) + "f}" + eenheid + " (%{customdata:,} vluchten)",
        ))
        if len(deel):
            eindlabel(fig, MAANDEN[int(deel["maand"].iloc[-1]) - 1], deel[kolom].iloc[-1], str(jaar), KLEUR_JAAR[jaar])
    opmaak(fig, y_titel=maat, hoogte=400, marge_rechts=60)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(categoryorder="array", categoryarray=MAANDEN)
    fig.add_shape(type="line", x0="mrt", x1="mrt", yref="paper", y0=0, y1=1, line=dict(color=INKT_ZACHT, width=1, dash="dot"))
    fig.add_annotation(x="mrt", yref="paper", y=1, text="16 mrt 2020: lockdown", showarrow=False, xanchor="left", xshift=6,
                       yanchor="top", font=dict(size=12, color=INKT_ZACHT))
    toon(fig)

    # --- Elke maand een stip: aantal vluchten tegen vertraging -----------------
    st.markdown("#### Drukte en vertraging per maand")
    fig = go.Figure()
    for jaar in (2019, 2020):
        deel = per_maand[per_maand["jaar"] == jaar]
        fig.add_trace(go.Scatter(
            x=deel["vluchten"], y=deel["aandeel"], mode="markers+text", name=str(jaar),
            text=[MAANDEN[m - 1] for m in deel["maand"]], textposition="top center", textfont=dict(size=11, color=INKT_ZACHT),
            marker=dict(size=11, color=KLEUR_JAAR[jaar], line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{text} " + str(jaar) + "</b><br>%{x:,} vluchten<br>%{y:,.1f}% vertraagd<extra></extra>",
        ))
    opmaak(fig, x_titel="Vluchten in die maand", y_titel="Aandeel vertraagd (%)", hoogte=440)
    toon(fig)

    def periode(maanden):
        deel = zonder_jaar[zonder_jaar["maand"].isin(maanden)].groupby("jaar").agg(n=("vertraagd", "size"), aandeel=("vertraagd", "mean"))
        return deel if {2019, 2020} <= set(deel.index) else None

    zomer, december = periode([7, 8, 9]), periode([12])
    tekst = "Elke stip is een maand. "
    if zomer is not None:
        tekst += (f"In juli t/m september ging het aandeel vertraagd van {nl(zomer.loc[2019, 'aandeel'] * 100)}% (2019) naar "
                  f"{nl(zomer.loc[2020, 'aandeel'] * 100)}% (2020), bij {nl(100 * zomer.loc[2020, 'n'] / zomer.loc[2019, 'n'])}% van de vluchten. ")
    if december is not None:
        tekst += (f"In december was het {nl(december.loc[2019, 'aandeel'] * 100)}% tegen {nl(december.loc[2020, 'aandeel'] * 100)}%, "
                  f"terwijl er maar {nl(100 * december.loc[2020, 'n'] / december.loc[2019, 'n'])}% van de vluchten was. ")
    if (zomer is not None and december is not None
            and zomer.loc[2020, "aandeel"] < 0.4 * zomer.loc[2019, "aandeel"]
            and december.loc[2020, "aandeel"] > 0.6 * december.loc[2019, "aandeel"]):
        tekst += ("In 2019 was de zomer de slechtste periode, in 2020 de beste. Minder drukte haalt dus de zomerpiek weg, "
                  "maar niet de vertraging in de winter: die heeft kennelijk een andere oorzaak dan drukte.")
    st.caption(tekst)

    # --- Welke bestemmingen bleven? -------------------------------------------
    st.markdown("#### Niet alleen minder vluchten, ook andere bestemmingen")
    na_maart = zonder_jaar[zonder_jaar["maand"] >= 4]
    per_best = na_maart.groupby(["gebied", "jaar"], observed=True).agg(
        vluchten=("vertraagd", "size"), bestemmingen=("Org/Des", "nunique")).reset_index()
    rijen = []
    for gebied in ["Europa en omgeving", "Intercontinentaal"]:
        deel = per_best[per_best["gebied"] == gebied].set_index("jaar")
        if {2019, 2020} <= set(deel.index):
            rijen.append({
                "Gebied": gebied,
                "Vluchten apr–dec 2019": int(deel.loc[2019, "vluchten"]), "Vluchten apr–dec 2020": int(deel.loc[2020, "vluchten"]),
                "Over (%)": round(100 * deel.loc[2020, "vluchten"] / deel.loc[2019, "vluchten"], 1),
                "Bestemmingen 2019": int(deel.loc[2019, "bestemmingen"]), "Bestemmingen 2020": int(deel.loc[2020, "bestemmingen"]),
            })
    if rijen:
        tabel(pd.DataFrame(rijen))
    st.caption("Vergeleken over april t/m december, de maanden na de lockdown. Op pagina 4 kun je de kaart kleuren op "
               "'Verandering aantal vluchten' om te zien welke bestemmingen het meest inleverden.")


# ==========================================================================
# 7 · VOORSPELLING
# ==========================================================================
def pagina_model():
    st.title("Kunnen we vertraging voorspellen?")
    selectie_regel(filters_gelden=False)
    st.markdown(
        f"Het model schat voor elke vlucht de **kans op {VERTRAAGD_VANAF_MIN} minuten of meer vertraging**, en daarnaast het "
        "verwachte aantal minuten. Het leert van een deel van de dagen en wordt getest op dagen die het nooit heeft gezien."
    )
    keuze = st.radio("Hoe testen we?", list(mdl.TESTMETHODEN), horizontal=True)
    r = getraind_model(mdl.TESTMETHODEN[keuze])
    test, ladder, basis = r["testset"], r["ladder"], r["basis"]
    eind = ladder.iloc[-1]

    a, b, c, d = st.columns(4)
    a.metric("Getraind op (vluchten)", nl(r["n_train"]), f"{r['dagen_train']} dagen", delta_color="off")
    b.metric("Getest op (vluchten)", nl(r["n_test"]), f"{r['dagen_test']} dagen", delta_color="off")
    c.metric("AUC (0,5 = gokken, 1 = perfect)", nl(eind["AUC"], 3))
    d.metric("Gemiddelde fout in minuten", nl(r["mae"], 1) + " min",
             f"{nl(r['mae'] - basis['mae'], 1)} min t.o.v. gokken", delta_color="inverse",
             help="Gokken = voor elke vlucht de mediaan van de trainingsdata voorspellen.")

    tab_bron, tab_goed, tab_fout, tab_zelf = st.tabs(
        ["Wat voegt elke bron toe?", "Hoe goed is het?", "Waar zit het ernaast?", "Zelf een vlucht invullen"])

    # ---------------- Wat voegt elke bron toe ----------------
    with tab_bron:
        links, rechts = st.columns(2)
        with links:
            st.markdown("##### Stap voor stap een bron erbij")
            fig = go.Figure(go.Scatter(
                x=ladder["Stap"], y=ladder["AUC"], mode="lines+markers+text", line=dict(color=BLAUW, width=2),
                marker=dict(size=10, line=dict(color="white", width=2)), text=[nl(w, 3) for w in ladder["AUC"]],
                textposition="top center", hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"))
            opmaak(fig, y_titel="AUC op de testdagen", hoogte=360, legenda=False)
            fig.update_yaxes(rangemode="normal", range=[ladder["AUC"].min() - 0.03, ladder["AUC"].max() + 0.03])
            fig.update_xaxes(tickangle=0)
            toon(fig)
            st.caption("AUC: de kans dat het model een vertraagde vlucht een hogere score geeft dan een vlucht die op tijd is. "
                       "Het rooster alleen doet al veel; elke bron voegt een stukje toe, de actuele situatie op de luchthaven het meest.")
        with rechts:
            st.markdown("##### Welke kenmerken wegen het zwaarst?")
            belang = r["belang"].head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=belang["Daling AUC"], y=belang["Kenmerk"], orientation="h", marker=dict(color=BLAUW),
                                   hovertemplate="<b>%{y}</b><br>AUC daalt met %{x:.3f} als je dit kenmerk door elkaar husselt<extra></extra>"))
            opmaak(fig, x_titel="Daling van de AUC", hoogte=360, legenda=False)
            fig.update_yaxes(showgrid=False)
            toon(fig)
            st.caption("Hoe langer de staaf, hoe slechter het model wordt als je dat kenmerk door elkaar husselt. Het dagweer staat laag. Dat betekent niet dat weer er niet toe doet: de baan die in gebruik is hangt af van de wind, "
                       "dus een deel van het weer zit al in 'Baan'. En een daggemiddelde mist de bui van één uur.")

    # ---------------- Hoe goed is het ----------------
    with tab_goed:
        links, rechts = st.columns(2)
        with links:
            st.markdown("##### Klopt een voorspelde kans van 40% ook echt?")
            kal = mdl.kalibratie(test)
            grootste = max(kal["voorspeld"].max(), kal["werkelijk"].max()) * 100 + 5
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, grootste], y=[0, grootste], mode="lines", name="Perfect",
                                     line=dict(color=INKT_ZACHT, width=1, dash="dot"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=kal["voorspeld"] * 100, y=kal["werkelijk"] * 100, mode="lines+markers", name="Model",
                                     line=dict(color=BLAUW, width=2), marker=dict(size=8, line=dict(color="white", width=2)),
                                     customdata=kal["vluchten"],
                                     hovertemplate="voorspeld %{x:,.1f}%<br>werkelijk %{y:,.1f}%<br>%{customdata:,} vluchten<extra></extra>"))
            opmaak(fig, x_titel="Voorspelde kans op vertraging (%)", y_titel="Werkelijk vertraagd (%)", hoogte=380)
            toon(fig)
            st.caption("De testvluchten zijn in tien even grote groepen verdeeld, van laagste naar hoogste voorspelde kans. "
                       "Ligt de blauwe lijn op de stippellijn, dan kun je de kans letterlijk nemen.")
        with rechts:
            st.markdown("##### Wat als het model 'vertraagd' moet roepen?")
            drempel = st.slider("Het model zegt 'vertraagd' vanaf een kans van (%)", 10, 80, 40, step=5) / 100
            v = mdl.verwarring(test, drempel)
            m1, m2 = st.columns(2)
            m1.metric("Terecht gewaarschuwd", nl(v["terecht_alarm"]))
            m2.metric("Vals alarm", nl(v["vals_alarm"]))
            m1.metric("Gemist (wel vertraagd)", nl(v["gemist"]))
            m2.metric("Terecht niets gezegd", nl(v["terecht_rustig"]))
            st.markdown(
                f"Van elke 100 waarschuwingen kloppen er **{nl(v['precisie'] * 100)}**. "
                f"Het model vindt **{nl(v['vangst'] * 100)}%** van alle vertraagde vluchten. "
                f"Wie nooit waarschuwt heeft in {nl((1 - basis['aandeel_vertraagd_test']) * 100)}% van de gevallen gelijk; "
                f"het model bij deze drempel in {nl(v['juist'] * 100)}%."
            )
            st.caption("Schuif de drempel omlaag: je vangt meer vertraagde vluchten, maar krijgt meer vals alarm. Er is geen drempel die beide oplost.")

        st.markdown("##### Het minutenmodel")
        st.markdown(
            f"Gemiddeld zit de voorspelling **{nl(r['mae'], 1)} minuten** naast de werkelijkheid. Wie voor elke vlucht de mediaan gokt, "
            f"zit er {nl(basis['mae'], 1)} minuten naast. Het model verklaart {nl(r['r2'] * 100)}% van de verschillen in vertraging tussen vluchten "
            f"(R² = {nl(r['r2'], 2)})."
        )

    # ---------------- Waar zit het ernaast ----------------
    with tab_fout:
        st.markdown("##### Per dag: voorspeld tegen werkelijk")
        per_dag = test.groupby("datum").agg(werkelijk=("vertraagd", "mean"), voorspeld=("kans", "mean"), vluchten=("kans", "size"),
                                            wpgt=("wpgt", "first")).reset_index()
        per_dag = per_dag[per_dag["vluchten"] >= 150]
        per_dag[["werkelijk", "voorspeld"]] *= 100
        per_dag["fout"] = per_dag["werkelijk"] - per_dag["voorspeld"]
        grootste = max(per_dag["werkelijk"].max(), per_dag["voorspeld"].max()) + 5
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, grootste], y=[0, grootste], mode="lines", name="Perfect",
                                 line=dict(color=INKT_ZACHT, width=1, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=per_dag["voorspeld"], y=per_dag["werkelijk"], mode="markers", name="Testdag",
            marker=dict(size=8, color=BLAUW, line=dict(color="white", width=1), opacity=0.85),
            customdata=hoverdata(per_dag.assign(dag=per_dag["datum"].dt.strftime("%d-%m-%Y")), ["dag", "vluchten", "wpgt"]),
            hovertemplate="<b>%{customdata[0]}</b><br>voorspeld %{x:,.1f}% · werkelijk %{y:,.1f}%<br>"
                          "%{customdata[1]} testvluchten · windstoten %{customdata[2]} km/u<extra></extra>"))
        for _, rij in per_dag.nlargest(3, "fout").iterrows():
            fig.add_annotation(x=rij["voorspeld"], y=rij["werkelijk"], text=rij["datum"].strftime("%d-%m-%Y"), showarrow=False,
                               yshift=12, font=dict(size=11, color=INKT))
        opmaak(fig, x_titel="Gemiddelde voorspelde kans die dag (%)", y_titel="Werkelijk vertraagd die dag (%)", hoogte=420)
        toon(fig)
        ergste = per_dag.nlargest(1, "fout").iloc[0]
        st.caption(
            f"Elke stip is een testdag met minstens 150 vluchten. Boven de stippellijn was het erger dan het model dacht. "
            f"De grootste misser: {ergste['datum'].strftime('%d-%m-%Y')}, voorspeld {nl(ergste['voorspeld'])}%, werkelijk {nl(ergste['werkelijk'])}%. "
            "Het niveau van een hele dag heeft het model goed te pakken, vooral doordat het weet hoe het uur ervoor verliep. "
            "De fout zit dus minder in de dag en meer in de losse vlucht."
        )

        links, rechts = st.columns(2)
        with links:
            st.markdown("##### Per uur van de dag")
            per_uur = test.groupby("uur").agg(werkelijk=("vertraagd", "mean"), voorspeld=("kans", "mean"), vluchten=("kans", "size")).reset_index()
            per_uur = per_uur[per_uur["vluchten"] >= 100]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=per_uur["uur"], y=per_uur["werkelijk"] * 100, mode="lines+markers", name="Werkelijk",
                                     line=dict(color=INKT_ZACHT, width=2), hovertemplate="%{y:,.1f}%"))
            fig.add_trace(go.Scatter(x=per_uur["uur"], y=per_uur["voorspeld"] * 100, mode="lines+markers", name="Voorspeld",
                                     line=dict(color=BLAUW, width=2, dash="dash"), hovertemplate="%{y:,.1f}%"))
            opmaak(fig, x_titel="Gepland uur", y_titel="Vertraagd (%)", hoogte=320)
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(dtick=2, ticksuffix=":00")
            toon(fig)
            verschil_uur = (per_uur["werkelijk"] - per_uur["voorspeld"]).abs() * 100
            st.caption(f"Het patroon over de dag volgt het model goed: het grootste verschil is {nl(verschil_uur.max(), 1)} procentpunt "
                       f"(om {int(per_uur.loc[verschil_uur.idxmax(), 'uur'])}:00).")
        with rechts:
            st.markdown("##### Hoe groter de vertraging, hoe vaker gemist?")
            groot = test.copy()
            groot["klasse"] = pd.cut(groot["vertraging_min"], [-2000, 0, 15, 30, 60, 5000], right=False,
                                     labels=["te vroeg", "0–15 min", "15–30 min", "30–60 min", "≥ 60 min"])
            per_grootte = groot.groupby("klasse", observed=True).agg(kans=("kans", "mean"), vluchten=("kans", "size")).reset_index()
            fig = go.Figure(go.Bar(x=per_grootte["klasse"].astype(str), y=per_grootte["kans"] * 100, marker=dict(color=BLAUW),
                                   text=[nl(w) + "%" for w in per_grootte["kans"] * 100], textposition="outside", cliponaxis=False,
                                   customdata=per_grootte["vluchten"],
                                   hovertemplate="%{x}<br>gemiddelde voorspelde kans %{y:,.1f}%<br>%{customdata:,} vluchten<extra></extra>"))
            opmaak(fig, x_titel="Werkelijke vertraging", y_titel="Gemiddelde voorspelde kans (%)", hoogte=320, legenda=False)
            fig.update_xaxes(showgrid=False)
            toon(fig)
            zwaar = test[test["vertraging_min"] >= 60]
            st.caption(f"Vluchten met een uur of meer vertraging kregen gemiddeld een kans van {nl(zwaar['kans'].mean() * 100)}%: "
                       "hoger dan gemiddeld, maar lang niet zeker. De zwaarste vertragingen ziet het model niet aankomen.")

        st.markdown("##### Beperkingen")
        st.markdown(
            "- **Dagweer is te grof.** Een bui of mist van een uur verdwijnt in het daggemiddelde. Weer per uur zou hier het meeste opleveren.\n"
            "- **De oorzaak zit vaak buiten de data.** Een toestel dat al te laat binnenkomt, een technisch probleem of een slot van de "
            "luchtverkeersleiding staat niet in onze kolommen.\n"
            "- **De baan en het weer zijn op de dag zelf bekend,** niet weken vooruit. Het model is dus vooral bruikbaar kort van tevoren.\n"
            "- **2020 is geen normaal jaar.** Kies hierboven 'Trainen op 2019, testen op 2020' om te zien hoe het model zich houdt in een jaar dat het niet kent."
        )
        if r["methode"] == "jaar":
            st.info(f"Getraind op 2019 voorspelt het model voor 2020 gemiddeld {nl(test['kans'].mean() * 100, 1)}% vertraagd; "
                    f"werkelijk was het {nl(test['vertraagd'].mean() * 100, 1)}%. In 2019 was dat {nl(basis['aandeel_vertraagd_train'] * 100, 1)}%. "
                    "Het model heeft dus uit 2019 geleerd dat minder drukte minder vertraging geeft, en past dat goed toe op het coronajaar.")

    # ---------------- Zelf een vlucht invullen ----------------
    with tab_zelf:
        st.markdown("Stel een vlucht samen en kijk wat het model ervan vindt.")
        bestemmingen = (df.dropna(subset=["Latitude"]).groupby("Org/Des", observed=True)
                        .agg(vluchten=("vertraagd", "size"), City=("City", "first"), Name=("Name", "first"),
                             Latitude=("Latitude", "first"), Longitude=("Longitude", "first"), afstand_km=("afstand_km", "first"))
                        .nlargest(60, "vluchten"))
        bestemmingen["label"] = bestemmingen["City"].astype(str) + " · " + bestemmingen["Name"].astype(str)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown("**De vlucht**")
            richting = st.radio("Richting", ["Vertrek", "Aankomst"], horizontal=True, key="zelf_richting")
            code = st.selectbox("Bestemming of herkomst", list(bestemmingen.index), format_func=lambda c: bestemmingen.loc[c, "label"])
            maatschappij = st.selectbox("Maatschappij", r["categorieen"]["maatschappij"][:25])
            toestel = st.selectbox("Vliegtuigtype", r["categorieen"]["ACT"][:25])
            baan = st.selectbox("Baan", sorted(k for k in r["categorieen"]["RWY"] if k != "Overig"), index=3)
        with k2:
            st.markdown("**Het moment**")
            uur = st.slider("Gepland uur", 6, 22, 17)
            weekdag = st.selectbox("Dag van de week", range(7), format_func=lambda i: WEEKDAGEN[i], index=4)
            maand = st.selectbox("Maand", range(1, 13), format_func=lambda i: MAANDEN[i - 1], index=6)
            drukte = st.slider("Vluchten in hetzelfde uur", 5, 65, 50)
            vluchten_dag = st.slider("Vluchten op die dag", 50, 760, 700, step=10)
        with k3:
            st.markdown("**Het weer en de situatie**")
            tavg = st.slider("Temperatuur (°C)", -5, 30, 18)
            prcp = st.slider("Neerslag (mm)", 0, 50, 0)
            wpgt = st.slider("Zwaarste windstoot (km/u)", 10, 105, 35)
            weet_actueel = st.checkbox("Ik weet hoe het het afgelopen uur ging", value=True)
            vorig = st.slider("Gemiddelde vertraging in het uur ervoor (min)", -10, 60, 10, disabled=not weet_actueel)

        vlucht = {
            "uur": uur, "weekdag": weekdag, "maand": maand, "drukte": drukte, "vluchten_dag": vluchten_dag,
            "richting": richting, "ACT": toestel, "RWY": baan, "maatschappij": maatschappij,
            "afstand_km": bestemmingen.loc[code, "afstand_km"], "Latitude": bestemmingen.loc[code, "Latitude"],
            "Longitude": bestemmingen.loc[code, "Longitude"],
            "tavg": tavg, "tmin": tavg - 5, "prcp": prcp, "wspd": wpgt / 3.5, "wpgt": wpgt, "pres": 1017,
            "vertraging_vorig_uur": vorig if weet_actueel else np.nan,
        }
        kans, minuten = mdl.voorspel_een_vlucht(r, vlucht)
        u1, u2, u3 = st.columns(3)
        u1.metric("Kans op vertraging (≥ 15 min)", nl(kans * 100) + "%",
                  f"{'+' if kans > basis['aandeel_vertraagd_train'] else ''}{nl((kans - basis['aandeel_vertraagd_train']) * 100)} %-punt t.o.v. gemiddeld",
                  delta_color="inverse")
        u2.metric("Verwachte vertraging", nl(minuten) + " min")
        u3.metric("Afstand", nl(vlucht["afstand_km"]) + " km")

        # Dezelfde vlucht op elk uur van de dag: zo zie je het effect van het tijdstip los van de rest
        uren = list(range(6, 23))
        kansen = [mdl.voorspel_een_vlucht(r, {**vlucht, "uur": u})[0] * 100 for u in uren]
        fig = go.Figure(go.Scatter(x=uren, y=kansen, mode="lines+markers", line=dict(color=BLAUW, width=2),
                                   marker=dict(size=7, line=dict(color="white", width=2)), hovertemplate="%{x}:00 → %{y:,.0f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=[uur], y=[kans * 100], mode="markers", marker=dict(size=14, color=ORANJE, line=dict(color="white", width=2)),
                                 hoverinfo="skip"))
        opmaak(fig, x_titel="Gepland uur (de rest van de vlucht blijft gelijk)", y_titel="Kans op vertraging (%)", hoogte=300, legenda=False)
        fig.update_xaxes(dtick=1, ticksuffix=":00")
        toon(fig)
        st.caption("De oranje stip is de vlucht die je hebt ingevuld. Minimumtemperatuur, windsnelheid en luchtdruk vult het dashboard zelf in "
                   "op basis van je keuzes.")


# --------------------------------------------------------------------------
# De gekozen pagina tonen
# --------------------------------------------------------------------------
{
    PAGINAS[0]: pagina_overzicht,
    PAGINAS[1]: pagina_inspectie,
    PAGINAS[2]: pagina_tijd,
    PAGINAS[3]: pagina_kaart,
    PAGINAS[4]: pagina_weer,
    PAGINAS[5]: pagina_jaren,
    PAGINAS[6]: pagina_model,
}[pagina]()
