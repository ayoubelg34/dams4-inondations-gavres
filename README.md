# DaMS4 Dash App — Inondations cotières à Gavres

Application Dash (Plotly) pour explorer les scénarios d’inondation côtière et
visualiser les séries temporelles (6h) + les indicateurs Smax.

## Lancer en local
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 dash_app.py
```

Par défaut, l’app écoute sur `0.0.0.0` et le port vient de `PORT` (ou `8050`).

## Données (mode complet vs mode démo)
L’app démarre même si les données lourdes sont absentes.
- `mon_dossier_csv_converti/` : séries temporelles (peut être lourd).
- `resume_Smax_final.csv` : features agrégées pour la prédiction.
- `Y_Smax.csv` : Smax réel.
- `best_model.joblib` : modèle ML (optionnel).

Si un fichier manque, l’UI affiche un message explicite et l’app passe en **mode démo**.

## Déploiement Render (recommandé)
Render ne supporte pas GitHub Pages pour une app Dash.

**Build Command**
```bash
pip install -r requirements.txt
```

**Start Command**
```bash
./start.sh
```

> Pensez à rendre le script exécutable : `chmod +x start.sh`

## Déploiement Hugging Face Spaces (Docker)
1. Créer un Space **Docker**.
2. Pousser ce repo avec `Dockerfile`.
3. HF détecte le port `7860` automatiquement.

Le container lance :
```
 gunicorn dash_app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
```

## Notes
- Le modèle **prévoit Smax** (surface maximale inondée).
- Le **dépassement de digue** est une règle physique.
- L’**inondation significative** est une interprétation opérationnelle basée sur le seuil.

## Pourquoi pas GitHub Pages ?
GitHub Pages ne supporte pas les apps Python server-side (Dash). Utilisez Render ou Hugging Face Spaces.
