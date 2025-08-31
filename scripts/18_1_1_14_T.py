"""
Script: ksh_lakasarak.py
Purpose:
    - Process housing price data from KSH (Hungarian Central Statistical Office).
    - Organize data into main categories (új/használt lakások, árak, nm² árak).
    - Clean and standardize numerical values.
    - Transform dataset into tidy format: ["Fő kategória", "Épülettípus", "Év", "Érték"].
"""

import pandas as pd
import numpy as np

# --- 1. Fő kategóriák definiálása ---
fő_kategóriák = [
    "Használt lakások ára, millió forint",
    "Új lakások ára, millió forint",
    "Használt lakások átlagos négyzetméterára, ezer forint",
    "Új lakások átlagos négyzetméterára, ezer forint"
]

# --- 2. "Fő kategória" oszlop létrehozása ---
dataset["Fő kategória"] = dataset["Épülettípus"].apply(
    lambda x: x if x in fő_kategóriák else np.nan
)
dataset["Fő kategória"] = dataset["Fő kategória"].ffill()

# --- 3. Szűrés: kiszűrjük a főkategória sorokat és az "Összesen" értékeket ---
dataset = dataset[~dataset["Épülettípus"].isin(fő_kategóriák)]
dataset = dataset[dataset["Épülettípus"] != "Összesen"]

# --- 4. Wide → Long formátum transzformáció ---
year_cols = [col for col in dataset.columns if col.isdigit()]
dataset = dataset.melt(
    id_vars=["Fő kategória", "Épülettípus"],
    value_vars=year_cols,
    var_name="Év",
    value_name="Érték"
)

# --- 5. Értékek tisztítása ---
dataset["Érték"] = dataset["Érték"].replace("..", np.nan)  # hiányzó értékek
dataset["Érték"] = (
    dataset["Érték"].astype(str)
    .str.strip()
    .str.replace(r'\.(?=\d{3})', '', regex=True)   # ezres pont eltávolítása (pl. 1.234 → 1234)
    .str.replace(',', '.', regex=False)            # tizedesvessző → tizedespont
)

# --- 6. Numerikus konverzió ---
dataset["Érték"] = pd.to_numeric(dataset["Érték"], errors='coerce')

# --- 7. Skálázás bizonyos kategóriáknál ---
def scale_value(row):
    """
    Ha az érték "millió forintban" van megadva (ár), akkor:
    - ha > 20, akkor feltételezzük, hogy ezer Ft helyett tízezerben van, osztjuk 10-zel
    Egyébként meghagyjuk az értéket változatlanul.
    """
    if row["Fő kategória"] in [
        "Új lakások ára, millió forint",
        "Használt lakások ára, millió forint"
    ]:
        return row["Érték"] / 10 if row["Érték"] > 20 else row["Érték"]
    return row["Érték"]

dataset["Érték"] = dataset.apply(scale_value, axis=1)

# --- 8. Év konverzió (dátum típusra) ---
dataset["Év"] = pd.to_datetime(dataset["Év"].astype(str) + "-01-01", errors='coerce')

# --- 9. Eredmény ---
print(dataset.head())  # első 5 sor ellenőrzéshez




