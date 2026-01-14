# Dash app DaMS4 - Inondations cotieres a Gavres

Application web interactive (Dash/Plotly) pour le projet universitaire DaMS4
(Theme 1 : inondations cotieres a Gavres, BRGM).

## Objectif
- Selectionner un scenario d'inondation.
- Visualiser les series temporelles d'entree (6h) pour les variables disponibles.
- Afficher Smax reel depuis `resume_Smax_final.csv` (si la colonne est presente).
- Afficher Smax predit si un metamodele est disponible.

## Lien avec le projet
L'application illustre le pipeline : representation parametrique des series ->
metamodele -> prediction de Smax. Elle permet de comparer un scenario reel et
sa prediction pour la surface maximale inondee.

## Lancer l'application
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python dash_app.py
```

## Options utiles
```bash
python dash_app.py \
  --scenarios_dir mon_dossier_csv_converti \
  --y_file resume_Smax_final.csv \
  --model_path best_model.joblib \
  --host 127.0.0.1 \
  --port 8050
```

## Notes sur Smax et le modele
- `resume_Smax_final.csv` doit contenir une colonne Smax (ex: `S_max` ou `Y_Smax`).
  Si la colonne est absente, l'application affiche "indisponible".
- Le modele est optionnel. Si `best_model.joblib` est present a la racine,
  il est charge automatiquement. Le fichier doit contenir `model`,
  `feature_names`, et `target`.
