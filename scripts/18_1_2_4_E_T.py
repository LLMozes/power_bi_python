"""
Script: ksh_epitesi_engedelyek.py
Purpose:
    - Fetch housing construction permit data from KSH (Hungarian Central Statistical Office).
    - Clean and reshape the dataset into tidy format:
      ["Régió", "Területi egység", "Év", "db"]

"""

import pandas as pd

# --- 1. Adatok letöltése a KSH weboldalról ---
url = "https://www.ksh.hu/stadat_files/lak/hu/lak0020.html"
tables = pd.read_html(url, encoding='latin1')  # minden táblát betölt
dataset = tables[0]

# --- 2. Csak az első 32 sor kell ---
dataset = dataset.head(32)

# --- 3. Kiszűrjük a felesleges fejléc sort ---
dataset = dataset[dataset.iloc[:, 0] != "Kiadott lakásépítési engedély"]

# --- 4. Oszlopok átnevezése ---
dataset.columns = (
    ["Területi egység", "Területi egység_1"] +
    [str(col[0]) if isinstance(col, tuple) else str(col) for col in dataset.columns[2:]]
)

# --- 5. Régió hozzárendelése ---
dataset["Régió"] = dataset["Területi egység"].where(
    dataset["Területi egység_1"].isin(["régió", "nagyrégióc"])
)
dataset["Régió"] = dataset["Régió"].bfill()  # a hiányzó régiókat kitöltjük lefelé

# --- 6. Kiszűrjük a régió/nagyrégió sorokat ---
dataset = dataset[~dataset["Területi egység_1"].isin(["régió", "nagyrégióc"])]
dataset = dataset.drop(columns=["Területi egység_1"])

# --- 7. Nem kívánt értékek kizárása ---
exclude_values = [
    "Alföld és Észak",
    "Ország összesen",
    "A létesítendõ lakóépületek száma"
]
dataset = dataset[~dataset["Területi egység"].isin(exclude_values)]

# --- 8. Oszlopok sorrendje ---
dataset = dataset[
    ["Régió", "Területi egység"] +
    [col for col in dataset.columns if col not in ["Régió", "Területi egység"]]
]

# --- 9. Wide → Long formátum transzformáció ---
value_vars = [col for col in dataset.columns if col.isdigit()]
dataset = dataset.melt(
    id_vars=["Régió", "Területi egység"],
    value_vars=value_vars,
    var_name="Év",
    value_name="db"
)

# --- 10. Oszlopok tisztítása és konverzió ---
dataset["Év"] = pd.to_datetime(dataset["Év"], format='%Y')
dataset["db"] = (
    dataset["db"].astype(str)
    .str.replace(" ", "")     # szóköz eltávolítása
)
dataset["db"] = pd.to_numeric(dataset["db"], errors='coerce').fillna(0).astype(int)

# --- 11. Eredmény ---
print(dataset.head(15))  # az első 15 sor ellenőrzéshez
