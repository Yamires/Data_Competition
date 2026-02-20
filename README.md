**Nom d’équipe:**ift3395_Massimo_Costarella-Serra_Yamir_Alejandro_Poldo_Silva
## Contenu 

1. **Prédiction avec Régression Logistique**  
   - Fichier de soumission : `ift3395_YamirPoldoSilvaV10.csv`  
   - Notebook associé : `notebook.ipynb`

2. **Prédiction avec Voting Classifier**  
   - Fichier de soumission : `ift3395_YAPS_MCS_V61.csv`  
   - Notebook associé : `notebook_apres_25.ipynb`

3. **Fonctions et utilitaires communs**  
   - Fichier Python contenant les fonctions partagées : `preprocessor_funct.py`

4. **Données brutes**  
   - `data/train.csv` : données d’entraînement  
   - `data/test.csv` : données de test  

## Instructions d’exécution

### 1. Créer un environnement virtuel Python
```bash
python -m venv venv
source venv/bin/activate   # sous macOS / Linux
venv\Scripts\activate      # sous Windows
```

### 2. Installer les dépendances
Assurez-vous d’avoir le fichier `requirements.txt`, puis exécutez :
```bash
pip install -r requirements.txt
```

### 3. Exécuter un notebook
Ouvrez le notebook de votre choix, puis lancez toutes les cellules :
- Pour la **régression logistique** → `notebook.ipynb`
- Pour le **Voting Classifier** → `notebook_apres_25.ipynb`

Les fichiers `.csv` de prédiction seront automatiquement générés à la fin de l’exécution.


