import pandas as pd

# Az URL, ahonnan az adatokat szeretnéd importálni
url = "https://www.ksh.hu/stadat_files/lak/hu/lak0014.html"

# Az adatok beolvasása egyedi karakterkódolással
tables = pd.read_html(url, encoding='latin1')  # 'latin1' az ISO-8859-1 megfelelője

# Az első tábla kiválasztása
dataset = tables[0]  # Az első HTML-tábla kiválasztása

# Az eredmény megjelenítése a terminálban
print("Az első tábla tartalma:")
print(dataset)