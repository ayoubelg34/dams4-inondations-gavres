import pandas as pd
import os
import glob

# --- CONFIGURATION ---
# Remplacez ceci par le chemin de votre dossier contenant les fichiers .dat
dossier_source = './CFMDG_XY_dataset' 

# Dossier où seront enregistrés les CSV (créé automatiquement s'il n'existe pas)
dossier_sortie = './mon_dossier_csv_converti'

# Noms des colonnes propres (pour éviter les problèmes d'espaces dans l'en-tête original)
column_names = [
    't_min', 'NM_m_IGN69', 'T_m_NM', 'S_m', 
    'Hs_m', 'Tp_s', 'Dp_deg', 'U_m_s', 'DU_deg'
]

# --- DÉBUT DU TRAITEMENT ---

# Création du dossier de sortie si nécessaire
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# Récupération de la liste de tous les fichiers .dat
fichiers_dat = glob.glob(os.path.join(dossier_source, '*.dat'))

print(f"{len(fichiers_dat)} fichiers trouvés. Début de la conversion...")

count = 0
for fichier in fichiers_dat:
    try:
        # 1. Lecture du fichier
        # skiprows=20 : saute les 20 premières lignes (métadonnées)
        # sep='\s+' : gère les espaces multiples comme séparateur
        df = pd.read_csv(fichier, skiprows=20, sep=r'\s+', names=column_names)
        
        # 2. Préparation du nom de sortie
        nom_fichier = os.path.basename(fichier) # ex: Dataset_X_sc_1.dat
        nom_csv = nom_fichier.replace('.dat', '.csv') # ex: Dataset_X_sc_1.csv
        chemin_sortie = os.path.join(dossier_sortie, nom_csv)
        
        # 3. Sauvegarde en CSV
        df.to_csv(chemin_sortie, index=False)
        
        print(f"✅ Converti : {nom_fichier}")
        count += 1
        
    except Exception as e:
        print(f"❌ Erreur sur le fichier {fichier} : {e}")

print(f"--- Terminé. {count} fichiers convertis dans '{dossier_sortie}' ---")