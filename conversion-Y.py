import pandas as pd

# Chemin de votre fichier Y
fichier_y_dat = './CFMDG_XY_dataset/Dataset_Y_sc_all.dat'
fichier_y_csv = 'Y_Smax.csv' # Le nom propre qu'on va lui donner

# 1. Lecture spécifique : On veut juste la ligne des surfaces
# On lit tout le fichier en tant que texte d'abord pour être flexible
with open(fichier_y_dat, 'r') as f:
    lines = f.readlines()

# La ligne qui contient les valeurs de Smax est la 5ème ligne (indice 4)
# Vérifiez visuellement si besoin, mais votre affichage montre que c'est bien la ligne 4
raw_values = lines[4].strip().split()

# 2. Conversion en DataFrame propre
# On crée une liste de scénarios qui correspond aux noms de vos fichiers X
# Attention : il faut que "Dataset_X_sc_1.csv" corresponde à la 1ère valeur
data = []
for i, val in enumerate(raw_values):
    num_scenario = i + 1 # Les scénarios commencent à 1
    
    data.append({
        'Nom_Fichier': f'Dataset_X_sc_{num_scenario}.csv', # La clé de jointure !
        'Y_Smax': float(val)
    })

df_y = pd.DataFrame(data)

# 3. Sauvegarde
df_y.to_csv(fichier_y_csv, index=False)

print(f"✅ Fichier Y réparé et converti : {fichier_y_csv}")
print(f"Nombre de scénarios trouvés : {len(df_y)}")
print(df_y.head())