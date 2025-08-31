"""
Script: ksh_lakas_adatok.py
Purpose:
    - Fetch housing data from the Hungarian Central Statistical Office (KSH) website
    - Clean and reshape the dataset
    - Transform into a long format (tidy data) with columns: 
      ["Megnevezés", "Jogi személy típusa", "Év", "Érték"]

"""

import pandas as pd

# --- 1. Adatok letöltése a KSH weboldalról ---
url = "https://www.ksh.hu/stadat_files/lak/hu/lak0012.html"
tables = pd.read_html(url, encoding='latin1')   # minden HTML táblát betölt
dataset = tables[0]                             # az első táblát választjuk

# --- 2. Levágjuk a táblát az "A lakás nagysága" sor előtt ---
cutoff_index = dataset[dataset.iloc[:, 0].str.contains("A lakás nagysága", na=False)].index[0]
dataset = dataset.iloc[:cutoff_index].reset_index(drop=True)

# --- 3. Adatok átalakítása: megnevezés hozzárendelése ---
megnevezes = None
rows = []

for _, row in dataset.iterrows():
    if pd.isna(row.iloc[1]):  
        # új "Megnevezés" blokk kezdődik
        megnevezes = row.iloc[0]
    elif "összesen" not in row.iloc[0].lower():  
        # az "összesen" sorokat kihagyjuk
        rows.append({
            "Megnevezés": megnevezes,
            "Jogi személy típusa": row.iloc[0],
            **row.iloc[1:].to_dict()
        })

# Új DataFrame a tisztított sorokból
dataset = pd.DataFrame(rows).fillna(0)

# --- 4. Átalakítás long formátumba (tidy data) ---
id_vars = ["Megnevezés", "Jogi személy típusa"]
year_columns = [col for col in dataset.columns if col.isdigit()]

dataset = dataset.melt(
    id_vars=id_vars,
    value_vars=year_columns,
    var_name="Év",
    value_name="Érték"
)

# --- 5. Oszlopok tisztítása és típuskonverzió ---
dataset["Év"] = pd.to_datetime(dataset["Év"], format='%Y')
dataset["Érték"] = pd.to_numeric(dataset["Érték"], errors='coerce').fillna(0).astype(int)

# --- 6. Eredmény ---
print(dataset.head())   # VS Code terminálban az első 5 sor
# dataset   # ha Jupyter Notebookban vagy Power BI Python scriptben fut


##print(dataset)  ha megszertnék jelníteni vs code ba is.