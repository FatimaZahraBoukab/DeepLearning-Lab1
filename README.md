# Lab 1 : Deep Learning avec PyTorch (Régression et Classification)

## 1. Objectif

Prendre en main PyTorch pour résoudre une tâche de **régression** et une tâche de **classification multiclasse** avec des réseaux de neurones MLP, en utilisant GridSearch (sklearn) pour choisir les hyperparamètres et plusieurs techniques de régularisation.

- **Notebook :** `Lab1_DeepLearning_PyTorch.ipynb` (exécuté sur Google Colab, GPU T4)
- **Données :** importées directement dans Colab avec `kagglehub` 

## 2. Outils

Python, PyTorch, scikit-learn, skorch (pont PyTorch / GridSearchCV), imbalanced-learn (SMOTE), pandas, matplotlib, seaborn.

---

## 3. Partie 1 : Régression (NYSE)

**Dataset :** [dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv`.
**Tâche :** prédire le prix de clôture (`close`) à partir de `open`, `low`, `high` et `volume`.

### Démarche

1. **EDA :** aucune valeur manquante ; `open`, `low`, `high` et `close` sont corrélés à 1,00 ; le volume est faiblement corrélé (-0,13) ; la distribution de `close` est très asymétrique avec de nombreuses valeurs extrêmes.
2. **Préparation :** échantillon aléatoire de 200 000 lignes, séparation 80/20, standardisation (scaler ajusté sur le train uniquement).
3. **Modèle :** MLP 4 → h1 → h2 → 1 avec ReLU, perte MSE.
4. **GridSearch** (sklearn, `cv=2`, score R², sur 40 000 lignes pour limiter le temps de calcul) sur le learning rate, l'optimiseur, le nombre d'epochs et la taille des couches.
   - Meilleurs paramètres : **Adam, lr = 0,001, 40 epochs, h1 = 32, h2 = 32** (R² CV = 0,9998).
5. **Courbes Loss/Epochs et R²/Epochs** (le R² joue le rôle de l'accuracy pour une régression) : la loss diminue et le R² dépasse 0,9998. Les courbes train et test sont superposées, donc il n'y a pas de sur-apprentissage. Les petites oscillations viennent du learning rate constant.
6. **Régularisation :** Dropout, L2 (weight decay), Early stopping, BatchNorm et une combinaison.

### Résultats (jeu de test)

| Modèle | Epochs | Loss test | R² test |
|---|---|---|---|
| **Base** | 40 | 0,000083 | **0,99991** |
| Dropout (0,2) | 40 | 0,012426 | 0,98663 |
| L2 (1e-3) | 40 | 0,000219 | 0,99976 |
| Early stopping | 26 | 0,000110 | 0,99988 |
| BatchNorm | 40 | 0,037222 | 0,95994 |
| Combiné (Dropout + L2 + Early stopping) | 6 | 0,000540 | 0,99942 |

### Interprétation

- Le modèle de base est déjà excellent et ne sur-apprend pas (écart train/test de l'ordre de 1e-6). **La régularisation n'apporte donc aucune amélioration.**
- Dropout et BatchNorm dégradent le résultat : ils ajoutent du bruit à une tâche de prédiction numérique précise, et BatchNorm est instable avec des prix très étendus.
- L'early stopping donne un résultat équivalent au modèle de base avec moins d'epochs (26 au lieu de 40).
- Le modèle combiné s'est arrêté après 6 epochs, car la patience (5) est trop courte pour une loss bruitée : il est sous-entraîné et sa comparaison n'est pas équitable.
- **Limites :** `close` est presque égal à `open`, `low` et `high` du même jour, ce qui rend la tâche facile. La séparation train/test est aléatoire et non chronologique, ce qui est optimiste pour des séries temporelles.

---

## 4. Partie 2 : Classification multiclasse (maintenance prédictive)

**Dataset :** [shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (10 000 lignes).
**Tâche :** prédire le type de panne (`Failure Type`, 6 classes).

### Démarche

1. **Pré-traitement :** pas de valeurs manquantes ni de doublons ; suppression des identifiants (`UDI`, `Product ID`) ; suppression de 9 lignes incohérentes (`Target = 1` avec « No Failure ») ; suppression de `Target` (version binaire de la cible, risque de fuite de données) ; encodage de `Type` et de la cible ; standardisation.
2. **EDA :** dataset très déséquilibré (9 652 « No Failure » contre 18 « Random Failures » avant nettoyage) ; corrélations fortes entre température de l'air et du process (0,88) et entre vitesse de rotation et torque (-0,87).
3. **Équilibrage :** séparation stratifiée 80/20 (7 992 train / 1 999 test), puis **SMOTE sur le train uniquement** (7 714 exemples par classe, 46 284 au total). Le test reste intact.
4. **Modèle :** MLP 6 → h1 → h2 → 6 avec ReLU, perte `CrossEntropyLoss`.
5. **GridSearch** (sklearn, `cv=3` stratifié, score **F1 macro**). SMOTE est placé dans un pipeline pour s'appliquer uniquement sur les données d'entraînement de chaque pli.
   - Meilleurs paramètres : **Adam, lr = 0,001, 60 epochs, h1 = 64, h2 = 32** (F1 macro CV = 0,38).
6. **Courbes Loss/Epochs et Accuracy/Epochs** : l'accuracy de train atteint 99 %, mais la loss de test remonte après environ 25 epochs, ce qui indique du **sur-apprentissage**.
7. **Métriques** (accuracy, sensibilité, précision, F1 par classe, matrices de confusion) sur le train et le test.
8. **Régularisation :** Dropout, L2, Early stopping, BatchNorm et une combinaison.

### Résultats du modèle de base (test)

| Classe | Recall | Exemples dans le test |
|---|---|---|
| Heat Dissipation Failure | 0,95 | 22 |
| No Failure | 0,94 | 1 929 |
| Overstrain Failure | 0,88 | 16 |
| Power Failure | 0,79 | 19 |
| Tool Wear Failure | 0,11 | 9 |
| Random Failures | 0,00 | 4 |

Accuracy 0,937, recall macro 0,61, F1 macro 0,52.

### Comparaison des modèles (jeu de test)

| Modèle | Epochs | Accuracy | Recall macro | F1 macro test | F1 macro train | Écart F1 |
|---|---|---|---|---|---|---|
| Base | 60 | 0,9370 | 0,6124 | 0,5204 | 0,6843 | 0,164 |
| Dropout | 60 | 0,8949 | 0,6577 | 0,4885 | 0,5545 | 0,066 |
| L2 | 60 | 0,8934 | 0,6418 | 0,4639 | 0,5518 | 0,088 |
| **Early stopping** | 43 | 0,9345 | 0,6488 | **0,5349** | 0,6633 | 0,128 |
| BatchNorm | 60 | 0,9405 | 0,5549 | 0,4842 | 0,7137 | 0,230 |
| Combiné | 66 | 0,9060 | 0,6574 | 0,5089 | 0,5766 | 0,068 |

### Interprétation

- **L'accuracy est trompeuse** : environ 96 % des lignes sont « No Failure ». Le F1 macro et le recall par classe sont plus fiables.
- Les pannes Heat Dissipation, Overstrain et Power sont bien détectées. **Random Failures et Tool Wear Failure sont très mal détectées**, et beaucoup de « No Failure » sont prédites comme Tool Wear.
- Dropout, L2 et le modèle combiné **réduisent l'écart train/test** (0,07 à 0,09 contre 0,16) et augmentent le recall macro, mais au prix de plus de fausses alarmes (accuracy de 0,89 à 0,91). **L'early stopping donne le meilleur F1 macro test (0,535)** avec moins d'epochs.
- BatchNorm augmente le sur-apprentissage (écart de 0,23).
- **Limites :** le test contient seulement 4 à 22 exemples par classe de panne, donc les écarts de F1 entre modèles (environ 0,03) restent dans la marge d'incertitude. L'early stopping surveille la loss de test ; avec un jeu de validation séparé, la méthode serait plus rigoureuse. Les exemples SMOTE sont synthétiques, ce qui favorise la mémorisation des classes rares.

---

## 5. Synthèse personnelle

Ce lab m'a permis de découvrir PyTorch et de construire des MLP pour la régression et la classification. J'ai appris à écrire ma propre boucle d'entraînement (perte, rétropropagation, suivi des courbes) et à utiliser GridSearchCV de sklearn avec des modèles PyTorch grâce à skorch.

J'ai aussi compris que **la régularisation n'est utile que lorsqu'il y a du sur-apprentissage** : elle n'a rien apporté sur la régression (déjà excellente) et n'a que partiellement aidé sur la classification. Enfin, avec des données déséquilibrées, l'accuracy est trompeuse : il faut utiliser le F1 macro et le recall par classe, appliquer SMOTE uniquement sur le train (et dans le pipeline du GridSearch) pour éviter la fuite de données, et rester prudent sur les classes rares évaluées sur très peu d'exemples. 
