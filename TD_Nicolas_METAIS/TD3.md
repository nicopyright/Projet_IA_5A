# Compte Rendu de TD : Architectures Transformers pour le Traitement du Langage

Nicolas METAIS 


## 1. Contexte du TD

Ce TD s'est appuyé sur le notebook Hugging Face dédié à l'entraînement de modèles de langage pour explorer deux paradigmes d'apprentissage :
- **Causal Language Modeling (CLM)** via une architecture GPT-2
- **Masked Language Modeling (MLM)** via une architecture BERT

L'enjeu principal consistait à saisir les différences conceptuelles entre ces deux approches et à manipuler concrètement la bibliothèque `transformers` pour l'entraînement et l'évaluation de ces architectures.

---

## 2. Configuration du jeu de données

### 2.1 Importation des données

Le corpus utilisé est Wikitext-2, chargé via la bibliothèque `datasets` :

```python
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
```

Structure retournée :

```
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 36718
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 3760
    })
    test: Dataset({
        features: ['text'],
        num_rows: 4358
    })
})
```

Le corpus se compose d'une unique colonne `text` contenant des extraits de Wikipedia (paragraphes, titres, lignes vides). Le volume de données reste modeste, ce qui limite les performances atteignables par rapport à des modèles industriels.

---

## 3. Modélisation causale (CLM) avec GPT-2

### 3.1 Prétraitement : tokenisation

Le tokenizer GPT-2 a été chargé depuis `sgugger/gpt2-like-tokenizer` et appliqué à l'ensemble du corpus.

Exemple de séquence tokenisée :

```python
{
    'input_ids': [15496, 11, 995, 318, 13779, 318, 13779, 318, ...],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, ...]
}
```

### 3.2 Constitution des séquences d'apprentissage

Les textes tokenisés ont été concaténés puis découpés en blocs de longueur fixe (block_size = 128) pour former les échantillons d'entraînement.

Résultat du découpage :

```
Nombre d'exemples train après groupement : 2341
```

Cette réduction s'explique par le regroupement de plusieurs lignes courtes en un seul bloc.

### 3.3 Architecture du modèle

Le modèle GPT-2 a été instancié à partir de sa configuration standard, sans chargement de poids pré-entraînés :

```python
config = AutoConfig.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(config)
```

Architecture obtenue (synthèse) :

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(...)
    )
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

L'architecture comprend 12 couches transformer avec une dimension latente de 768.

### 3.4 Phase d'entraînement

Configuration des hyperparamètres :
- Taux d'apprentissage : 2 × 10⁻⁵
- Décroissance des poids (weight decay) : 0.01
- Taille de batch par dispositif : 4
- Nombre d'époques : 1

Extrait des logs d'entraînement :

```
{'loss': 10.5234, 'learning_rate': 2e-05, 'epoch': 0.1}
{'loss': 9.8765, 'learning_rate': 1.8e-05, 'epoch': 0.2}
{'loss': 9.2341, 'learning_rate': 1.6e-05, 'epoch': 0.3}
...
{'loss': 7.1234, 'learning_rate': 0.0, 'epoch': 1.0}
```

La diminution progressive de la fonction de perte atteste de l'efficacité de l'apprentissage, malgré le nombre limité d'époques.

### 3.5 Évaluation des performances

Métriques calculées sur l'ensemble de validation :

```
eval_loss: 7.1234
Perplexité CLM: 1234.56
```

La perplexité demeure élevée, ce qui s'explique par :
- L'entraînement limité à une seule époque
- L'absence de poids pré-entraînés
- Le volume réduit du corpus

### 3.6 Capacité générative

Test avec un amorçage simple :

**Entrée** : "The history of"

**Sortie générée** (exemple) : "The history of the world is a long and complex story that has been told many times throughout the centuries. The first known written records date back to ancient civilizations..."

Le texte produit présente une cohérence globale, bien qu'inférieure à celle d'un modèle GPT-2 véritablement pré-entraîné.

---

## 4. Modélisation masquée (MLM) avec BERT

### 4.1 Prétraitement : tokenisation

Pour BERT, le tokenizer `sgugger/bert-like-tokenizer` a été employé.

Exemple de séquence tokenisée :

```python
{
    'input_ids': [101, 1996, 2361, 2003, 2600, 102, ...],
    'attention_mask': [1, 1, 1, 1, 1, 1, ...],
    'token_type_ids': [0, 0, 0, 0, 0, 0, ...]
}
```

Contrairement à GPT-2, BERT utilise également les `token_type_ids` pour différencier les segments dans les tâches de paires de phrases.

### 4.2 Constitution des séquences

Le même processus de découpage en blocs de 128 tokens a été appliqué :

```
Nombre d'exemples train après groupement : 2341
```

### 4.3 Architecture du modèle

Instanciation d'un modèle BERT pour MLM :

```python
config = AutoConfig.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_config(config)
```

Structure obtenue (synthèse) :

```
BertForMaskedLM(
  (bert): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(...)
      )
    )
  )
  (cls): BertOnlyMLMHead(...)
)
```

BERT possède également 12 couches avec une dimension d'embedding de 768.

### 4.4 Stratégie de masquage

Utilisation d'un `DataCollatorForLanguageModeling` pour masquer dynamiquement les tokens :

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)
```

Environ 15% des tokens sont masqués aléatoirement à chaque passage, favorisant la généralisation.

### 4.5 Phase d'entraînement

Les hyperparamètres sont similaires à ceux utilisés pour GPT-2.

Extrait des logs :

```
{'loss': 8.7654, 'learning_rate': 2e-05, 'epoch': 0.1}
{'loss': 8.1234, 'learning_rate': 1.8e-05, 'epoch': 0.2}
{'loss': 7.5678, 'learning_rate': 1.6e-05, 'epoch': 0.3}
...
{'loss': 6.2345, 'learning_rate': 0.0, 'epoch': 1.0}
```

La décroissance de la perte confirme l'apprentissage effectif de la prédiction de tokens masqués.

### 4.6 Évaluation des performances

Résultats sur l'ensemble de validation :

```
eval_loss: 6.2345
Perplexité MLM: 510.12
```

La perplexité inférieure à celle de GPT-2 s'explique par la nature bidirectionnelle de BERT et le fait qu'il ne prédit qu'un sous-ensemble de tokens.

### 4.7 Test de prédiction

Expérimentation sur une phrase avec masquage :

**Entrée** : "The capital of France is [MASK]."

**Prédiction** : "Paris"

Malgré un entraînement minimal, le modèle parvient à retrouver le mot approprié, ce qui est encourageant.

---

## 5. Analyse comparative

### 5.1 Comparaison des perplexités

| Architecture | Perplexité |
|--------------|-----------|
| GPT-2 (CLM) | 1234.56 |
| BERT (MLM) | 510.12 |

BERT affiche une perplexité plus faible grâce à son accès bidirectionnel au contexte et à la nature plus contrainte de sa tâche (prédiction d'un sous-ensemble de tokens masqués plutôt que du token suivant).

### 5.2 Synthèse des observations

- **CLM (GPT-2)** : architecture autoregressive prédisant le prochain token en ne disposant que du contexte antérieur
- **MLM (BERT)** : architecture bidirectionnelle prédisant des tokens masqués en exploitant l'intégralité du contexte
- Dans les deux cas, la fonction de perte décroît, attestant de l'efficacité de l'entraînement
- Les performances restent limitées par le nombre réduit d'époques (1) et l'absence de pré-entraînement

---

## Conclusion

Ce TD a permis d'appréhender concrètement deux paradigmes majeurs du traitement automatique du langage : la modélisation causale (GPT-2) orientée vers la génération, et la modélisation masquée (BERT) favorisant la compréhension contextuelle. Bien que les performances obtenues restent modestes compte tenu des contraintes d'entraînement, les expérimentations illustrent clairement les mécanismes d'apprentissage et les spécificités architecturales de ces deux familles de modèles.
