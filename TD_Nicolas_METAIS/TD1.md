# Compte Rendu - TD1 : Apprentissage du Perceptron

Nicolas Metais



## 1. Introduction

Ce TD a pour objectif d'implémenter et d'étudier l'apprentissage d'un perceptron simple. Le perceptron est un classificateur linéaire qui peut apprendre à séparer des données linéairement séparables.

### 1.1 Objectifs

- Comprendre le fonctionnement d'un perceptron simple
- Implémenter l'algorithme d'apprentissage par la règle de delta
- Apprendre la fonction logique ET (AND) à partir d'exemples
- Analyser la convergence de l'algorithme d'apprentissage





## 2. Méthodologie

### 2.1 Architecture du perceptron

Le perceptron implémenté possède :
- **2 entrées** : x₁ et x₂ (valeurs binaires 0 ou 1)
- **1 biais** : w₀ (fixé à -1)
- **2 poids synaptiques** : w₁ et w₂
- **1 sortie** : fonction sigmoïde de la somme pondérée

### 2.2 Fonction d'activation

La fonction sigmoïde est utilisée comme fonction d'activation :
```
σ(x) = 1 / (1 + e^(-x))
```

Cette fonction permet d'obtenir une sortie continue entre 0 et 1, ce qui facilite l'apprentissage par gradient.

### 2.3 Algorithme d'apprentissage

L'algorithme utilisé est la règle de delta  :

Pour chaque exemple d'apprentissage (x, d) :
1. Calculer la sortie : y = σ(w₀·bias + w₁·x₁ + w₂·x₂)
2. Calculer l'erreur : δ = d - y
3. Mettre à jour les poids :
   - w₀ = w₀ + α · bias · δ
   - w₁ = w₁ + α · x₁ · δ
   - w₂ = w₂ + α · x₂ · δ

où α = 0.7 est le taux d'apprentissage.

### 2.4 Données d'apprentissage

Le perceptron apprend la fonction logique ET avec les 4 exemples possibles :

| x₁ | x₂ | Sortie attendue |
|----|----|----------------|
| 0  | 0  | 0              |
| 0  | 1  | 0              |
| 1  | 0  | 0              |
| 1  | 1  | 1              |

### 2.5 Paramètres expérimentaux

- **Nombre d'itérations** : 1000
- **Taux d'apprentissage** : 0.7
- **Initialisation des poids** : aléatoire entre -2 et 0
- **Fonction d'activation** : sigmoïde



## 3. Résultats

### 3.1 Convergence de l'algorithme

L'algorithme converge généralement en quelques centaines d'itérations. La courbe d'erreur montre une décroissance rapide au début, puis une stabilisation autour d'une valeur proche de zéro.

### 3.2 Poids appris

Après l'apprentissage, le perceptron a appris des poids qui permettent de séparer correctement les exemples. Les valeurs exactes dépendent de l'initialisation aléatoire, mais le perceptron parvient toujours à apprendre la fonction ET.

### 3.3 Performance

Le perceptron entraîné est capable de classer correctement les 4 exemples d'apprentissage avec une précision de 100% pour la fonction logique ET.





## 4 Analyse des résultats

Le perceptron simple est capable d'apprendre la fonction logique ET car cette fonction est **linéairement séparable**. Il existe un hyperplan (dans ce cas, une droite) qui peut séparer les points de classe 0 des points de classe 1.

### 4.1 Limitations

Le perceptron simple présente des limitations importantes :
- Il ne peut apprendre que des fonctions linéairement séparables
- Il ne peut pas apprendre la fonction OU exclusif (XOR), qui n'est pas linéairement séparable
- La convergence n'est pas garantie si les données ne sont pas linéairement séparables

### 4.2 Influence des paramètres

- **Taux d'apprentissage** : Un taux trop élevé peut provoquer des oscillations, un taux trop faible ralentit la convergence
- **Initialisation** : L'initialisation des poids influence le temps de convergence mais pas le résultat final (pour des données linéairement séparables)

---

## 5. Conclusion

Ce TD a permis de mettre en pratique les concepts fondamentaux du perceptron et de comprendre son fonctionnement. L'implémentation réussit à apprendre la fonction logique ET, démontrant la capacité du perceptron à résoudre des problèmes linéairement séparables.



