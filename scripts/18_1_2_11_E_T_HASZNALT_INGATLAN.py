import pandas as pd
url = "https://www.ksh.hu/stadat_files/lak/hu/lak0027.html"
tables = pd.read_html(url, encoding='latin1')
dataset = tables[0]
dataset = dataset[1:].reset_index(drop=True)
column_names = ["Régió", "Épülettípus"] + list(dataset.columns[2:])
dataset.columns = column_names
orszag_index = dataset[dataset["Régió"] == "Ország"].index
if not orszag_index.empty:
    dataset = dataset.loc[:orszag_index[0] - 1]
dataset = dataset[~dataset["Épülettípus"].str.contains("együtt", na=False)]
year_columns = [col for col in dataset.columns if col.isdigit()]
non_year_columns = ["Régió", "Épülettípus"]
dataset = dataset.melt(
    id_vars=non_year_columns, 
    var_name="Év", 
    value_name="MillióFt")

dataset["Év"] = pd.to_datetime(dataset["Év"], format='%Y', errors='coerce')
dataset["MillióFt"] = dataset["MillióFt"].str.replace(",", ".", regex=False)  
dataset["MillióFt"] = pd.to_numeric(dataset["MillióFt"], errors='coerce') / 10 
dataset["MillióFt"] = dataset["MillióFt"].map(lambda x: f"{x:.1f}".replace(".", ","))
dataset


##print(dataset)  ha megszertnék jelníteni vs code ba is.