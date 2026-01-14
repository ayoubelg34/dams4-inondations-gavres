import pandas as pd

# 1. Chargement du fichier Y (celui que vous venez de vérifier)
# Assurez-vous d'avoir sauvegardé vos données Y dans un fichier 'Y_Smax.csv'
df_y = pd.read_csv('Y_Smax.csv') 

# 2. Chargement du fichier X (celui généré à l'étape d'avant avec t=0)
df_x = pd.read_csv('Dataset_X_Features.csv')

# 3. FUSION (MERGE)
# On utilise la colonne 'Nom_Fichier' comme clé commune
df_final = pd.merge(df_x, df_y, on='Nom_Fichier')

# 4. Vérification
print("--- Résumé du Dataset Final ---")
print(f"Nombre de lignes : {len(df_final)}")
print(f"Nombre de colonnes : {len(df_final.columns)}")
print("\nAperçu des colonnes :")
print(df_final.columns.tolist())

# 5. Sauvegarde pour le Machine Learning
df_final.to_csv('Dataset_Complet.csv', index=False)
print("\n✅ C'est gagné ! Le fichier 'Dataset_Complet.csv' est prêt pour le LASSO et le Random Forest.")