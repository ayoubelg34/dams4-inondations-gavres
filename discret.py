import pandas as pd
import os
import glob

# --- CONFIGURATION ---
dossier_source = './CFMDG_XY_dataset' 
dossier_sortie = './mon_dossier_csv_converti'
fichier_final_X = 'Dataset_X_Features.csv' # Le fichier récapitulatif

# Noms des colonnes
column_names = [
    't_min', 'NM_m_IGN69', 'T_m_NM', 'S_m', 
    'Hs_m', 'Tp_s', 'Dp_deg', 'U_m_s', 'DU_deg'
]

# --- DÉBUT DU TRAITEMENT ---

if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

fichiers_dat = glob.glob(os.path.join(dossier_source, '*.dat'))
print(f"{len(fichiers_dat)} fichiers trouvés. Extraction des Max/Moyennes en cours...")

data_features = []

count = 0
for fichier in fichiers_dat:
    try:
        # 1. Lecture du fichier .dat
        df = pd.read_csv(fichier, skiprows=20, sep=r'\s+', names=column_names)
        
        # 2. Sauvegarde en CSV (Optionnel, pour garder une trace propre)
        nom_fichier_dat = os.path.basename(fichier)
        nom_csv = nom_fichier_dat.replace('.dat', '.csv')
        df.to_csv(os.path.join(dossier_sortie, nom_csv), index=False)
        
        # 3. CALCUL DES PARAMÈTRES (Votre demande)
        # Moyenne pour NM et U, Max pour le reste
        features = {
            'Nom_Fichier': nom_csv, # Clé pour la fusion future
            
            # --- Les Moyennes ---
            'NM_mean': df['NM_m_IGN69'].mean(), # Niveau Moyen
            'U_mean':  df['U_m_s'].mean(),      # Vent
            
            # --- Les Max ---
            'T_max':   df['T_m_NM'].max(),      # Marée Haute Max
            'S_max_input': df['S_m'].max(),     # Surcote Max (Attention: ne pas confondre avec Y_Smax)
            'Hs_max':  df['Hs_m'].max(),        # Vagues Max
            'Tp_max':  df['Tp_s'].max(),        # Période Max
            'Dp_max':  df['Dp_deg'].max(),      # Direction vagues (Max)
            'DU_max':  df['DU_deg'].max(),      # Direction vent (Max)
            
            # --- Feature Engineering Bonus ---
            # Somme des pires cas (Max Marée + Max Surcote + Moyenne Niveau)
            # C'est une estimation du "Niveau Total Max Potentiel"
            'Niveau_Total_Estime': df['NM_m_IGN69'].mean() + df['T_m_NM'].max() + df['S_m'].max()
        }
        
        data_features.append(features)
        count += 1
        
    except Exception as e:
        print(f"❌ Erreur sur le fichier {fichier} : {e}")

# 4. Création du tableau final X
df_X = pd.DataFrame(data_features)
df_X = df_X.sort_values('Nom_Fichier')

# 5. Sauvegarde
df_X.to_csv(fichier_final_X, index=False)

print(f"✅ Terminé ! {count} fichiers traités.")
print(f"📊 Le fichier de paramètres '{fichier_final_X}' a été créé.")
print("Aperçu :")
print(df_X.head())