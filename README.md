# Compétition 1 : Prédiction de la Qualité de la Bière 🍺

Bienvenue à votre première compétition d'intelligence artificielle ! Dans ce défi, vous travaillerez avec un ensemble de données contenant des **échantillons de bière** et certaines de leurs **propriétés chimiques**. Votre objectif est d'entraîner un modèle de **classification** pour prédire la **qualité** de chaque bière en fonction de ces propriétés.

## Introduction

La bière est l'une des boissons alcoolisées les plus consommées au monde depuis près de 4 000 ans. Cette boisson est obtenue à partir de la fermentation d'amidons, principalement dérivés de céréales telles que l'orge maltée, le blé, le maïs et le riz. Le processus de brassage comprend plusieurs étapes, notamment le maltage, l'ébullition, la fermentation, le conditionnement et l'emballage. Chaque étape peut influencer de manière significative la saveur, l'arôme et la qualité globale du produit final.

## Énoncé du Problème

### Description de l'Ensemble de Données

Vous disposez d'un ensemble de données d'échantillons de bière où chaque échantillon est caractérisé par :
- **Attributs chimiques** : Diverses propriétés chimiques mesurables de la bière, catégorielles ou numériques (bitterness_IBU, beer_style, diacetyl_concentration...)
- **Score de qualité** : Une variable cible **entière** représentant la note de qualité globale de la boisson

### Formulation Mathématique

Formalisons ceci comme un **problème d'apprentissage supervisé** :

- **Caractéristiques d'entrée** : $\mathbf{x} = [x_1, x_2, ..., x_d] \in \mathbb{R}^d$ où $d$ est le nombre d'attributs chimiques
- **Variable cible** : $y \in \{1, 2, 3, ..., 10\}$ représentant des scores de qualité **discrets** de 1 (mauvais) à 10 (excellent)
- **Ensemble d'entraînement** : $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ avec $n$ échantillons d'entraînement

⚠️ Il s'agit d'un **problème de classification multi-classes** où nous visons à prédire le score de qualité, comme une **classe discrète**, et non une valeur continue. ⚠️

**Métrique d'Évaluation** : La performance de votre modèle sera évaluée en utilisant la **précision**, définie comme :

$$\text{Précision} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}_{[\hat{y}_i = y_i]}$$

où $\mathbb{1}_{[\hat{y}_i = y_i]}$ est la fonction indicatrice qui vaut 1 lorsque la prédiction $\hat{y}_i$ correspond à l'étiquette réelle $y_i$, et 0 sinon.

### Format de la Compétition 🏆

Cette compétition suit un **système d'évaluation en deux phases** : Vous soumettrez vos prédictions sur [Kaggle](https://www.kaggle.com/competitions/ift-6390-ift-3395-beer-quality-prediction/). Votre modèle sera évalué sur l'ensemble de test fourni (test.csv). Cet ensemble de test est divisé en deux parties : un **ensemble de test public** et un **ensemble de test privé**. Vous pouvez faire jusqu'à 3 soumissions par jour.

**Classement Public** : Votre performance sur l'ensemble de test public sera affichée à tout moment sur le tableau de classement public, ce qui vous donne un retour immédiat sur les performances de votre modèle. Utilisez cela pour itérer et améliorer votre approche.

**Classement Privé** : Votre performance sur l'ensemble de test privé déterminera votre classement final dans la compétition. Le classement sur le dataset privé est caché pendant la compétition pour éviter le surapprentissage et affiché uniquement à la fin de la compétition.

**Important** : Votre score final sera déterminé par la performance sur l'**ensemble de test privé** uniquement, alors concentrez-vous sur la construction de modèles robustes et généralisables plutôt que sur l'optimisation uniquement pour le tableau de classement public !

## Dates Limites Importantes

- **25 octobre 2025** : Date limite pour battre les différentes baselines en utilisant **uniquement les méthodes vues en classe**
- **8 novembre 2025** : Date limite pour soumettre votre meilleur modèle et compétitionnez les uns contre les autres

## Pour Commencer
Avant de commencer à explorer les données et à construire vos modèles, vous devrez configurer votre environnement Python avec les dépendances requises.

### 📋 Configuration de l'Environnement

Assurez-vous d'avoir Python 3.9 ou supérieur installé ainsi que pip. Vous pouvez utiliser soit Conda soit Virtual Environment (venv) pour gérer vos dépendances.

#### Option 1 : Utilisation de Conda 🐍
```bash
# Créer un nouvel environnement conda
conda create -n .venv python=3.9

# Activer l'environnement
conda activate .venv

# Installer les dépendances
pip install -r requirements.txt
```

#### Option 2 : Utilisation de Virtual Environment (venv) 📦
```bash
# Créer un environnement virtuel
python3.9 -m venv .venv

# Activer l'environnement
# Sur Linux/macOS :
source .venv/bin/activate
# Sur Windows :
# .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 📁 Structure du Projet

```
Competition_1/
├── README.md                  # Ce fichier
├── requirements.txt           # Paquets requis
├── notebook.ipynb             # Notebook principal pour entraîner votre modèle
├── data/                      # Dossier de l'ensemble de données
│   │── train.csv              # Données d'entraînement au format csv
│   │── test.csv               # Données de test au format csv
│   └── sample_submission.csv  # exemple de soumission
```

## Votre Mission 

1. **Explorer** les ensembles de données d'entraînement et de test et comprendre les propriétés de chaque attribut (données manquantes, caractéristiques redondantes ou déséquilibrées)
2. **Prétraiter** les données (mise à l'échelle, normalisation, encodage des caractéristiques non numériques, etc. soyez créatifs !)
3. **Entraîner** divers modèles d'apprentissage automatique vus en classe
4. **Évaluer** la performance du modèle en utilisant des métriques appropriées dans un rapport
5. **Générer** des prédictions pour l'ensemble de test et les soumettre sur Kaggle

### Exploration et Visualisation des Données

La première étape de tout projet d'apprentissage automatique est de charger et d'examiner les données. Vous pouvez **par exemple** :

- Vérifier les types de caractéristiques (numériques, catégorielles) et les valeurs min/max
- Visualiser la distribution de certaines caractéristiques, distributions conditionnelles, corrélations, etc.
- Vérifier visuellement les valeurs aberrantes ou anomalies, et la séparabilité linéaire des classes

### Prétraitement des Données

Selon la méthode que vous choisissez, nous pourrions avoir besoin de prétraiter nos données (nettoyage, transformation, etc.). Voici quelques étapes de prétraitement courantes que vous pourriez vouloir mettre en œuvre :

1. **Gestion des Valeurs Manquantes** : Identifier et gérer de manière appropriée toutes les valeurs manquantes dans l'ensemble de données, soit par imputation soit par suppression.
2. **Encodage des Variables Catégorielles** : Convertir les variables catégorielles en format numérique en utilisant des techniques telles que l'encodage one-hot ou l'encodage d'étiquettes.
3. **Mise à l'Échelle des Caractéristiques** : Normaliser ou standardiser les caractéristiques numériques pour s'assurer qu'elles sont sur une échelle similaire, ce qui est crucial pour de nombreux algorithmes d'apprentissage automatique.
4. **Sélection de Caractéristiques** : Identifier et conserver les caractéristiques les plus pertinentes qui contribuent de manière significative à la tâche de prédiction, en utilisant potentiellement des techniques comme l'analyse de corrélation ou l'importance des caractéristiques à partir de modèles.
5. **Toutes étapes de prétraitement supplémentaires que vous souhaitez essayer, soyez créatifs !**

### Division Entraînement et Validation

Quelle que soit l'approche, nous vous suggérons de toujours mettre de côté des données de l'ensemble d'entraînement pour évaluer la performance de votre modèle et effectuer le réglage des hyperparamètres :

$$\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{val}$$

Où :
- $\mathcal{D}_{train}$ : Utilisé pour l'entraînement des modèles (80% des données)
- $\mathcal{D}_{val}$ : Utilisé pour l'évaluation et la sélection de modèle (20% des données)

### Entraînement et Évaluation du Modèle

Entraînez plusieurs modèles d'apprentissage automatique et comparez leurs performances. Chaque modèle apprend un type différent de frontière de décision.

Une fois que vous avez entraîné et évalué différents modèles, vous devriez choisir le meilleur et l'analyser plus en détail. Voici quelques suggestions sur ce qu'il faut inclure dans votre analyse :

1. **Performance du Modèle** : Rapportez les métriques de performance (précision, rappel, score F1) sur les ensembles d'entraînement et de validation. Discutez de toute divergence entre les performances d'entraînement et de validation.
2. **Matrice de Confusion** : Présentez une matrice de confusion pour visualiser la performance du modèle sur différentes classes de qualité. Identifiez quelles classes sont le plus fréquemment mal classées.
3. **Importance des Caractéristiques** : Si applicable, analysez l'importance des différentes caractéristiques dans le modèle. Vous pouvez par exemple discuter quelles propriétés chimiques sont les plus influentes pour prédire la qualité de la bière.
4. **Analyse des Erreurs** : Enquêtez sur les instances où le modèle a fait des prédictions incorrectes. Recherchez des motifs dans les mauvaises classifications et considérez les raisons potentielles de ces erreurs.

## Instructions de Soumission

### Format de Soumission Kaggle

Votre fichier de soumission doit suivre le format de `sample_submission.csv`, qui est un fichier CSV avec deux colonnes et une ligne d'en-tête :
- **id** : L'identifiant de chaque échantillon de test
- **quality** : Votre score de qualité prédit (entier de 1 à 10)

Exemple de format :
```
id,quality
1,7
2,5
3,8
...
```

Assurez-vous que vos prédictions sont des entiers dans l'intervalle [1, 10] et que vous incluez des prédictions pour tous les échantillons de test.

## Formation d'Équipe et Configuration Kaggle

### Exigences de Taille d'Équipe
- **Étudiants IFT3395** : Peuvent travailler en équipes de **jusqu'à 2 étudiants**
- **Étudiants IFT6390** : Doivent travailler **individuellement**

### Format du Nom d'Équipe Kaggle

Le nom de votre équipe Kaggle **doit** suivre ce format :

**Pour les équipes IFT3395 :**
- Équipe de 1 : `ift3395_prenom_nom`
- Équipe de 2 : `ift3395_prenom1_nom1_prenom2_nom2`

**Pour les étudiants IFT6390 :**
- `ift6390_prenom_nom`

**Exemples :**
- `ift3395_mehdi-inane_ahmed`
- `ift3395_mehdi-inane_ahmed_tom_marty`
- `ift6390_charlie_tremblay`

⚠️ **Important** : L'utilisation du format de nommage correct est obligatoire pour une attribution appropriée des notes.

## Exigences du Rapport

En plus de mettre en œuvre votre méthode, vous devez rédiger un court rapport détaillant votre méthodologie pour résoudre ce problème et présentant les résultats de votre méthode. Plus précisément, votre rapport doit contenir les informations suivantes :

1. **Nom d'Équipe Kaggle** (suivant le format spécifié ci-dessus : `ift3395_...` ou `ift6390_...`), ainsi que la liste des membres de l'équipe (nom complet et numéro d'étudiant)
2. **Prétraitement des Caractéristiques (Conception des Caractéristiques)** : Décrivez et justifiez vos étapes de prétraitement des caractéristiques et indiquez quelles caractéristiques vous avez sélectionnées pour votre modèle.
3. **Méthodologie** : Décrivez et justifiez toutes les décisions concernant la division des données en ensembles d'entraînement et de validation, ainsi que les techniques utilisées pour améliorer la performance (stratégie de régularisation, réglage des hyperparamètres, etc.)
4. **Résultats** : Présentez une analyse concise de vos résultats à l'aide de tableaux ou de graphiques.
5. **Discussion** : Commentez vos résultats et indiquez les avantages et les inconvénients de votre approche et méthodologie.

Le rapport ne doit pas dépasser **2 pages**. Vous êtes libre de structurer le rapport comme vous le souhaitez tant que vous incluez les éléments mentionnés ci-dessus. Les sections introduction, description du problème et conclusion ne sont pas obligatoires.

### Éléments à Soumettre

Soumettez les éléments suivants sur **Gradescope** (le lien sera fourni sur Piazza) avant la date limite finale (5 novembre 2025) :
1. **Rapport** (format PDF, maximum 2 pages)
2. **Code** (notebook Jupyter ou scripts Python)

⚠️ **Important** : Assurez-vous que le nom de votre équipe Kaggle correspond à celui que vous indiquez dans votre rapport !

## Notation (Total 100 points)

### **Compétition de Données (60 points)**
- **20 points** : Battre la baseline aléatoire
- **30 points** : Battre la baseline forte (Cela devrait être réalisé uniquement en utilisant les méthodes vues en classe)
- **5 points** : Se classer au-dessus de la performance médiane
- **5 points** : Atteindre le top 3

**Note** : Les prédictions des baselines sont disponibles dans la section `Leaderboard` sur Kaggle.

### **Rapport Écrit (30 points)**
- **6 points** : Format et présentation
- **8 points** : Description et justification des algorithmes
- **8 points** : Méthodologie et conception expérimentale
- **8 points** : Discussion des résultats et analyse

### **Soumission du Code (10 points)**
- **2 points** : Code bien commenté
- **4 points** : Lisibilité et organisation du code
- **4 points** : Documentation sur la façon d'exécuter le code (README, instructions, etc.)

## Conseils pour Réussir

- Commencez tôt
- N'oubliez pas l'exploration des données, le prétraitement et l'ingénierie des caractéristiques (c'est la partie la plus importante)
- Essayez différents modèles et comparez leurs performances
- Documentez votre approche et vos découvertes
- Amusez-vous et apprenez quelque chose de nouveau !

## Règles de la Compétition

L'objectif de cette compétition est de vous donner l'opportunité d'apprendre les aspects clés et les subtilités de ce qui fait une bonne méthode de classification, de l'analyse des données, du prétraitement, à l'entraînement du modèle et à la sélection des hyperparamètres.

- Utilisez l'IA générative de manière responsable, votre classement ne compte que pour 10% de la note totale !
- Toutes les règles de l'UdeM sur le plagiat s'appliquent.

---

Bonne chance pour la compétition ! 🏅
