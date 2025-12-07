# Compte Rendu de Travaux Dirigés : Réseaux Antagonistes Génératifs (GANs)

Nicolas METAIS

---

## 1. Présentation générale

Ce TD explore les réseaux antagonistes génératifs (GANs) à travers deux approches complémentaires :
- **GAN Lab**, plateforme interactive permettant une exploration pédagogique des mécanismes internes d'un GAN
- **StyleGAN3**, architecture avancée pour la génération d'images photoréalistes haute résolution

L'objectif global consistait à développer une double compréhension, théorique et pratique, du fonctionnement des GANs. GAN Lab offre un environnement visuel simplifié où observer la dynamique d'apprentissage et les interactions entre générateur et discriminateur. À l'opposé, StyleGAN3 illustre les défis techniques des modèles état de l'art, particulièrement en termes d'exigences matérielles.

---

## 2. Exploration via GAN Lab

### 2.1 Principes fondamentaux du modèle

GAN Lab permet d'étudier le comportement d'un GAN dans un contexte réduit où les données sont représentées par des distributions de points bidimensionnels. Cette simplification rend observables les dynamiques d'apprentissage adversarial, habituellement difficiles à appréhender dans les générateurs d'images complexes. L'interface permet un suivi en temps réel des ajustements du générateur et des réactions du discriminateur, rendant le processus d'entraînement beaucoup plus accessible.

#### 2.1.1 Structure architecturale

La compréhension des visualisations de GAN Lab nécessite de rappeler l'architecture d'un GAN, composé de deux réseaux neuronaux en opposition :
- **Générateur** : transforme des vecteurs aléatoires (bruit latent) en échantillons artificiels
- **Discriminateur** : évalue l'authenticité des données, distinguant productions synthétiques et données réelles

Ces réseaux évoluent dans une dynamique antagoniste : chaque amélioration du générateur oblige le discriminateur à affiner sa capacité de détection. Cette compétition produit idéalement un équilibre où le générateur reproduit fidèlement la distribution cible.

#### 2.1.2 Outils de visualisation

GAN Lab propose plusieurs interfaces visuelles pour rendre transparents les mécanismes d'apprentissage :

1. **Model Overview Graph** : représentation schématique de l'architecture et des flux de données
2. **Layered Distributions View** : superposition visuelle des distributions réelle, générée et évaluée par le discriminateur

Ces éléments révèlent non seulement l'état instantané du modèle, mais aussi l'évolution temporelle des distributions.

#### 2.1.3 Représentations avancées

L'outil intègre également des visualisations détaillées :
- **Manifold** : transformation progressive des vecteurs de bruit en échantillons
- **Heatmap du discriminateur** : cartographie spatiale de la frontière décisionnelle
- **Gradients** : directions d'amélioration suggérées pour les échantillons

Ces visualisations explicitent l'effet des mises à jour paramétriques et permettent de comprendre les phénomènes d'oscillation ou d'effondrement modal (mode collapse).

### 2.2 Expérimentation paramétrique

L'exploration dans GAN Lab consiste à varier les hyperparamètres pour observer leur impact sur la stabilité et la convergence. L'équilibre entre les deux réseaux étant fragile, des modifications minimes peuvent induire des comportements radicalement différents.

#### 2.2.1 Vitesse d'apprentissage du générateur

**Configurations testées** : 0.0001, 0.001, 0.01

Observations :
- **0.0001** : convergence extrêmement lente vers la distribution cible
- **0.001** : apprentissage équilibré avec stabilité acceptable
- **0.01** : progression rapide mais instable, risque d'effondrement modal

Ces résultats soulignent l'importance du taux d'apprentissage : une valeur excessive empêche le générateur d'exploiter efficacement les retours du discriminateur.

#### 2.2.2 Vitesse d'apprentissage du discriminateur

**Configurations testées** : 0.0001, 0.001, 0.01

Comportements analogues :
- **0.0001** : discriminateur insuffisamment réactif
- **0.001** : évolution harmonieuse des deux réseaux
- **0.01** : discriminateur trop performant bloquant l'amélioration du générateur

Un discriminateur surpuissant cesse de fournir un gradient exploitable, rejetant catégoriquement les échantillons artificiels sans nuance.

#### 2.2.3 Fréquence d'entraînement du discriminateur

**Valeurs testées** : 1, 2, 3, 5 itérations par mise à jour du générateur

Impact observé :
- **1 itération** : calibration insuffisante du discriminateur
- **2–3 itérations** : équilibre satisfaisant entre les réseaux
- **5 itérations** : dominance excessive du discriminateur

Ce paramètre, souvent négligé, s'avère crucial pour prévenir les dérives d'apprentissage.

#### 2.2.4 Dimensionnalité de l'espace latent

**Plage explorée** : 10 à 200 dimensions

Constats :
- **10–20** : diversité limitée des échantillons générés
- **50–100** : couverture adéquate de la distribution réelle
- **200+** : complexité accrue sans bénéfice substantiel

Un espace latent restreint limite la variabilité des générations, tandis qu'une dimensionnalité excessive complique l'optimisation.

#### 2.2.5 Type de distribution du bruit

La comparaison entre distributions uniforme et gaussienne confirme que la distribution normale facilite les interpolations et s'harmonise mieux avec la géométrie naturelle de l'espace latent employé par les GANs contemporains.

---

## 3. StyleGAN 3 : Génération d'images avec réseaux antagonistes génératifs

### 3.1 Introduction à StyleGAN 3

StyleGAN 3, développé par NVIDIA Research, représente une avancée majeure dans la génération d'images par réseaux antagonistes génératifs (GAN). Contrairement aux réseaux de classification étudiés précédemment, les GAN sont conçus pour **générer** de nouvelles données plutôt que de les classifier.

**Principales innovations de StyleGAN 3** :
- **Équivariance aux translations** : Les détails générés ne sont plus fixés à des coordonnées spécifiques
- **Réduction de l'aliasing** : Élimination des artefacts visuels présents dans StyleGAN 2
- **Cohérence spatiale améliorée** : Les transformations géométriques sont mieux gérées
- **Architecture alias-free** : Conception mathématiquement rigoureuse pour éviter les problèmes d'échantillonnage

### 3.2 Architecture de StyleGAN 3

#### 3.2.1 Composants principaux

**1. Mapping Network (Réseau de mapping)**
- Transforme un vecteur latent aléatoire `z` en un vecteur de style `w`
- Architecture : MLP avec 8 couches
- Objectif : Décorréler les caractéristiques dans l'espace latent

**2. Synthesis Network (Réseau de synthèse)**
- Génère l'image à partir du vecteur de style `w`
- Architecture progressive : commence à 4×4 et monte jusqu'à la résolution cible
- Utilise la modulation de style (AdaIN) pour contrôler les caractéristiques

**3. Discriminator (Discriminateur)**
- Distingue les images réelles des images générées
- Entraîné de manière antagoniste avec le générateur

#### 3.2.2 Améliorations par rapport à StyleGAN 2

| Aspect | StyleGAN 2 | StyleGAN 3 |
|--------|------------|------------|
| Aliasing | Présent | Éliminé |
| Équivariance | Partielle | Complète |
| Stabilité | Bonne | Excellente |
| Cohérence spatiale | Moyenne | Élevée |

### 3.3 Implémentation et résultats


**Impossibilité de générer des images réelles** : Malgré les efforts déployés pour implémenter et tester StyleGAN 3, il n'a pas été possible de générer des images réelles avec un modèle pré-entraîné. Les raisons principales sont :

1. **Ressources matérielles insuffisantes** : L'entraînement ou l'utilisation d'un modèle StyleGAN 3 pré-entraîné nécessite un GPU avec au moins 8GB de VRAM, ce qui n'était pas disponible dans notre environnement de travail.

2. **Complexité de l'installation** : L'implémentation officielle de NVIDIA nécessite des dépendances spécifiques (PyTorch avec CUDA, bibliothèques spécialisées) qui n'ont pas pu être correctement configurées dans notre environnement.

3. **Taille des modèles pré-entraînés** : Les modèles pré-entraînés disponibles font plusieurs centaines de mégaoctets, et leur téléchargement et utilisation nécessitent des ressources réseau et de stockage importantes.

4. **Temps de calcul** : Même avec un modèle pré-entraîné, la génération d'images sur CPU prendrait plusieurs minutes par image, ce qui n'était pas pratique pour nos tests.



## 4. Conclusion

Ce TD a permis de mettre en pratique les concepts avancés des réseaux de neurones multicouches, à la fois pour la classification et pour la génération d'images. 
Ces travaux illustrent la polyvalence et la puissance des réseaux de neurones profonds dans différents domaines de l'intelligence artificielle.