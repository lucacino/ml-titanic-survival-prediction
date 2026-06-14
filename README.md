# Titanic Survival Prediction 🚢

> Predicting passenger survival using a Decision Tree classifier — with a focus on preventing data leakage and selecting the optimal model depth.

**IT** | Progetto di Machine Learning realizzato nell'ambito del Master in Data Science, Analytics e AI @ Start2Impact University.

---

## Obiettivo / Objective

**IT:** Prevedere la sopravvivenza di un passeggero del Titanic a partire da variabili demografiche e di viaggio, confrontando più profondità dell'albero decisionale per individuare il modello con il miglior equilibrio bias-varianza.

**EN:** Predict whether a Titanic passenger survived, using demographic and travel features. The core challenge: find the Decision Tree depth that maximizes generalization without overfitting.

---

## Dataset

| Feature | Description |
|---|---|
| `Sex` | Gender of the passenger |
| `Age` | Age (177 missing values → imputed with median) |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Embarked` | Port of embarkation (2 missing values → imputed with mode) |
| `Survived` | Target variable (0 = No, 1 = Yes) |

Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic) — 891 passengers

---

## Metodologia / Methodology

### Split dei Dati
```
Dataset completo (891)
├── Train Set (668) → Train Finale (501) + Validation Set (167)
└── Test Set (223) → usato una sola volta per la valutazione finale
```
`random_state=0` su tutti gli split per garantire la riproducibilità.

Il preprocessing è stato calcolato **solo sul training set** e poi applicato su validation e test — per evitare il data leakage.

### Preprocessing
- `Age` → imputazione con la **mediana** (distribuzione simmetrica, robusta agli outlier)
- `Embarked` → imputazione con la **moda** (variabile categorica)
- `Sex` → **LabelEncoder** (variabile binaria: male=1, female=0)
- `Embarked` → **OneHotEncoder** (variabile nominale → colonne `Embarked_Q`, `Embarked_S`)

### Selezione della Profondità

| max_depth | Accuracy Train | Accuracy Validation |
|---|---|---|
| 2 | 79.04% | 79.04% |
| **5** | **85.23%** | **80.24%** ✅ |
| 10 | 90.62% | 78.44% |
| 25 | 91.22% | 77.84% |
| None | 91.22% | 77.84% |

`depth=5` selezionato: massima accuracy di validation, nessun segno di overfitting.

---

## Risultati / Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **81.17%** |
| Baseline (majority class) | 61.38% |
| Miglioramento vs baseline | +19.79 pp |

Confusion Matrix (Test Set):
- 127 non sopravvissuti identificati correttamente (recall 91%)
- 54 sopravvissuti identificati correttamente (recall 64%)
- 30 falsi negativi (sopravvissuti predetti come non sopravvissuti)

---

## Analisi Esplorativa — Insight chiave

- Sesso: feature più discriminante — donne 74.2% sopravvivenza vs uomini 18.9%
- Classe di viaggio: forte correlazione — 63% in 1ª classe, 24.2% in 3ª
- Dataset sbilanciato (61.6% / 38.4%): accuracy da sola non sufficiente → valutare precision e recall

---

## Tech Stack

- Python — pandas, numpy, scikit-learn, matplotlib
- Modello: `DecisionTreeClassifier` (scikit-learn)
- Ambiente: Jupyter Notebook

---

## Struttura del Repository

```
ml-titanic-survival-prediction/
├── README.md
├── titanic_survival_prediction.ipynb
├── data/
│   └── titanic.csv
└── presentation/
    └── MACHINE_LEARNING.pdf
```

---

## Come Riprodurre / How to Run

```bash
git clone https://github.com/lucacino/ml-titanic-survival-prediction
cd ml-titanic-survival-prediction
pip install -r requirements.txt
jupyter notebook titanic_survival_prediction.ipynb
```

---

## Autore / Author

**Luca Cino** — [LinkedIn](https://www.linkedin.com/in/lucacino) | [GitHub](https://github.com/lucacino)

Master Professionale in Data Science, Analytics e AI @ Start2Impact University
