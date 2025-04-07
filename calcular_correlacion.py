import pandas as pd

# cargar el archivo de Excel
archivo = "IADataSet.xlsx"
df = pd.read_excel(archivo, sheet_name = "Datas")

# Opcional: Verifica los nombres de columnas
print("Columnas disponibles:", df.columns.to_list())

# Nombres exactos de las columnas
columna_x = 'active_minutes'
columna_y = 'calories_burned'

# Eliminar valores nulos
df = df[[columna_x, columna_y]].dropna()

# Calcular correlación de Pearson
correlacion = df.corr().iloc[0,1]
print(f"Correlacion entre {columna_x} y {columna_y}: {correlacion:.2f}")