# Compétition de données — Prédiction de la qualité de la bière

Ce projet vise à prédire la qualité d’une bière à partir de ses caractéristiques. Le notebook prépare les données, compare plusieurs modèles et crée un fichier de prédiction pour la compétition.

**Nom d’équipe :** `ift3395_Massimo_Costarella-Serra_Yamir_Alejandro_Poldo_Silva`

## Contenu

| Fichier | Description |
| --- | --- |
| [notebookv1.ipynb](notebookv1.ipynb) | Notebook de préparation des données, d’entraînement et de prédiction. |
| [notebookv2.pdf](notebookv2.pdf) | Version PDF d’un notebook. |
| [preprocessor_funct.py](preprocessor_funct.py) | Fonctions de préparation des données et d’évaluation des modèles. |
| [requirements.txt](requirements.txt) | Dépendances Python. |
| [data/train.csv](data/train.csv) | Données d’entraînement, avec la colonne cible `quality`. |
| [data/test.csv](data/test.csv) | Données de test pour les prédictions. |
| [predictionV1.csv](predictionV1.csv), [predictionv2.csv](predictionv2.csv) | Fichiers de prédiction déjà présents dans le dépôt. |

## Modèles

Le notebook compare la régression logistique, le Gradient Boosting et Extra Trees. Il combine ensuite leurs probabilités avec un **Voting Classifier** et ajuste les poids des classes 4 et 8.

Les modèles sont évalués sur un ensemble de validation, notamment avec le rappel macro.

## Instructions d’exécution

Exécutez les commandes depuis le dossier contenant ce README.

### 1. Créer un environnement virtuel

```bash
python -m venv venv
```

Activez-le sous macOS ou Linux :

```bash
source venv/bin/activate
```

Sous Windows, dans l’invite de commandes :

```bat
venv\Scripts\activate
```

### 2. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### 3. Ouvrir le notebook

Ouvrez `notebookv1.ipynb` dans VS Code ou Jupyter, sélectionnez l’environnement virtuel et exécutez les cellules dans l’ordre depuis le dossier du projet.

