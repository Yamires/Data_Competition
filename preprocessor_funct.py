from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score
)
import matplotlib.pyplot as plt

def high_corr(df: pd.DataFrame, t: float) -> list:
    """
    Fonction pour identifier les caractéristiques hautement corrélées.
        Args: df (pd.DataFrame): Le DataFrame contenant les données.
            t (float): Le seuil de corrélation au-dessus duquel les caractéristiques sont considérées comme hautement corrélées.
        Returns: list: Une liste des noms des caractéristiques hautement corrélées.
    """
    # Calculer la matrice de corrélation absolue
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool))
    # Trouver les caractéristiques avec une corrélation supérieure au seuil
    return [c for c in corr_matrix.columns if any(upper[c] > t)]


def low_corr(df: pd.DataFrame, t: float) -> list:
    """
    Fonction pour identifier les caractéristiques faiblement corrélées avec la cible 'quality'.
        Args: df (pd.DataFrame): Le DataFrame contenant les données.
            t (float): Le seuil de corrélation en dessous duquel les caractéristiques 
                                sont considérées comme faiblement corrélées.
        Returns:list: Une liste des noms des caractéristiques faiblement corrélées.
    """
    # Vérifier si la colonne 'quality' existe dans le DataFrame
    if 'quality' not in df:
        return []
    # Calculer la corrélation absolue avec la cible 'quality'
    c = df.corr(numeric_only=True)['quality'].abs()
    # Trouver les caractéristiques avec une corrélation inférieure au seuil
    return c[c < t].index.tolist()
   
def drop_by_correlation(df: pd.DataFrame, high_threshold=0.9, low_threshold=0.05) -> tuple[pd.DataFrame, list]:
    """
    Fonction pour supprimer les caractéristiques basées sur la corrélation.
        Args: df (pd.DataFrame): Le DataFrame contenant les données.
            high_corr_threshold (float)
            low_corr_threshold (float)
        Returns: pd.DataFrame: Le DataFrame avec les caractéristiques supprimées.
                list: Une liste des noms des caractéristiques supprimées.
    """
    # Identifier les caractéristiques à faible et haute corrélation
    low = low_corr(df, low_threshold)
    high = high_corr(df.drop(columns=['quality'], errors='ignore'), high_threshold)
    
    # Supprimer les caractéristiques identifiées
    features_to_drop = list(set(low + high))
    
    df = df.drop(columns=features_to_drop, errors='ignore')

    print("Features supprimées :", features_to_drop)

    return df, features_to_drop


def impute_values(df: pd.DataFrame,  is_train: True, stats: Optional[Dict[str, Any]] = None)  -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """ 
    Fonction pour imputer les valeurs manquantes dans les colonnes 'lactic_acid' et 'sodium'.
        Args: df (pd.DataFrame): Le DataFrame contenant les données.
            is_train (bool): Indique si le DataFrame est pour l'entraînement ou le test.
            stats (Optional[Dict[str, Any]]): Statistiques d'imputation pour le jeu de test.
        Returns: pd.DataFrame: Le DataFrame avec les valeurs imputées.
                Optional[Dict[str, Any]]: Statistiques d'imputation pour le jeu d'entraînement.
    """
    dF = df.copy()
    
    cols = ['lactic_acid', 'sodium']

    for col in cols:
            dF.loc[dF[col] < 0, col] = np.nan
   
    if is_train:

        mean = {}
        
        global_means = dF[cols].mean().to_dict()
        mean['global_means'] = global_means
        
        dF = dF.fillna(dF.groupby('quality')[cols].transform('mean'))
        dF = dF.fillna(global_means)
        
        return dF, mean
        
        
    dF = dF.fillna(stats['global_means'])
    
    return dF, None
    

def gscore(y_val, y_pred_final, model):
    print(f"Accuracy: {accuracy_score(y_val, y_pred_final):.4f}")
    print(f"Recall Macro: {recall_score(y_val, y_pred_final, average='macro'):.4f}")
    print(f"F1 Macro: {f1_score(y_val, y_pred_final, average='macro'):.4f}")

    print("\n" + classification_report(y_val, y_pred_final))

    cm = confusion_matrix(y_val, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.tight_layout()
    plt.show()

def training_score(model, y_val, y_pred):
    print(f"Meilleurs hyperparam {model.__class__.__name__} :{model.best_params_}")
    print(f"Meilleur score en CV: {model.best_score_}")

    print(f"Accuracy sur validation: {accuracy_score(y_val, y_pred)}")
    print(f"Recall macro sur validation: {recall_score(y_val, y_pred, average="macro")}")



def features_engenering(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    """
    Fonction de prétraitement pour créer de nouvelles caractéristiques basées sur les caractéristiques existantes.
        Args:    processed_df (pd.DataFrame): Le DataFrame contenant les données prétraitées.
        Returns:    pd.DataFrame: Le DataFrame avec les nouvelles caractéristiques ajoutées.
    """
    
    df = df.copy()

    df['free_CO2/dissolvedO2'] = df['free_CO2'] / (df['dissolved_oxygen'])
    df['gypmsum_level_x_sodium'] = df['gypsum_level'] * df['sodium']
    df['fermentation_strenght/alcohol_ABV'] = df['fermentation_strength'] /(df['alcohol_ABV'])
    df['acid_x_gas'] = ((df['lactic_acid']) / (df['pH'])) * df['free_CO2/dissolvedO2']
    df['log_gravity_drop'] = np.sqrt((df['original_gravity'] - df['final_gravity']).clip(lower=0))
    df['bitterness_alcohol_ratio'] = df['bitterness_IBU'] / (df['alcohol_ABV'])
        
    return df

def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Évalue les performances d’un modèle de classification

    Args:
        model : object
        X : pd.DataFrame 
        y : pd.Series 
    Returns:
        dict : 'Acc', 'Macro_Recall', 'Macro_F1', 'Macro_Precision'
    """

    # Calcule des prédiction 
    proba = model.predict_proba(X)
    preds = model.classes_[proba.argmax(axis=1)]

    # Retourne un dictionnaire contenant plusieurs métriques globales
    return {
        "Acc": accuracy_score(y, preds),                          
        "Macro_Recall": recall_score(y, preds, average="macro"),   
        "Macro_F1": f1_score(y, preds, average="macro"),         
        "Macro_Precision": precision_score(y, preds, average="macro"),
    }
