# 🐧 Classificació de Pingüins de Palmer

Projecte de Machine Learning per classificar espècies de pingüins (Adelie, Chinstrap, Gentoo) utilitzant 4 models diferents: Regressió Logística, SVM, Arbres de Decisió i KNN.

## 📑 Taula de Continguts

- [Descripció del Projecte](#-descripció-del-projecte)
- [Instal·lació i Configuració](#-installació-i-configuració)
- [Estructura del Projecte](#-estructura-del-projecte)
- [Executar el Servidor Flask](#-executar-el-servidor-flask)
- [API Endpoints](#-api-endpoints)
- [Executar el Client de Prova](#-executar-el-client-de-prova)
- [Desenvolupament](#-desenvolupament)
- [Exemples de Predicció](#-exemples-de-predicció)
- [Gestió d'Errors](#-gestió-derrors)
- [Resolució de Problemes](#-resolució-de-problemes)
- [Requisits del Projecte](#-requisits-del-projecte)
- [Llicència](#-llicència)
- [Enllaços](#-enllaços)

## 📋 Descripció del Projecte

Aquest projecte implementa una API REST amb Flask que serveix 4 models de classificació entrenats amb el dataset Palmer Penguins. Els models poden predir l'espècie d'un pingüí basant-se en característiques físiques i geogràfiques.

### Models Implementats

- **Regressió Logística** (`logistic_regression`)
- **Support Vector Machine** (`svm`)
- **Arbre de Decisió** (`decision_tree`)
- **K-Nearest Neighbors** (`knn`)

### Variables Predictores

- `island`: Illa on es va observar el pingüí (Torgersen, Biscoe, Dream)
- `bill_length_mm`: Longitud del bec en mm
- `bill_depth_mm`: Profunditat del bec en mm
- `flipper_length_mm`: Longitud de l'aleta en mm
- `body_mass_g`: Massa corporal en grams
- `sex`: Sexe del pingüí (Male, Female)

## 🚀 Instal·lació i Configuració

### Prerequisits

- Conda (Anaconda o Miniconda)
- Git
- Python 3.10

### Pas 1: Clonar el Repositori

```bash
git clone https://github.com/jcortesa/IABD_25-26_CE_5073-3.1-tasca-avaluable.git
cd IABD_25-26_CE_5073-3.1-tasca-avaluable
```

### Pas 2: Crear l'Entorn Conda

```bash
# Crear entorn amb Python 3.10
conda create --name penguins-classification python=3.10

# Activar l'entorn
conda activate penguins-classification
```

### Pas 3: Instal·lar Dependències

**Opció recomanada** - Utilitzar el fitxer `environment.yml` per garantir la reproductibilitat:

```bash
conda env create -f environment.yml
conda activate penguins-classification
```

**Alternativament**, pots instal·lar les dependències manualment:

```bash
# Activar l'entorn
conda activate penguins-classification

# Instal·lar totes les dependències necessàries
conda install scikit-learn pandas seaborn matplotlib flask jupyter requests ipykernel
```

## 📂 Estructura del Projecte

```
IABD_25-26_CE_5073-3.1-tasca-avaluable/
│
├── datasets/
│   └── penguins.csv              # Dataset Palmer Penguins
│
├── notebooks/
│   └── penguins_classification.ipynb  # Notebook amb EDA i entrenament
│
├── models/
│   ├── logistic_regression.pck   # Model Regressió Logística
│   ├── svm.pck                   # Model SVM
│   ├── decision_tree.pck         # Model Arbre de Decisió
│   ├── knn.pck                   # Model KNN
│   ├── dict_vectorizer.pck       # Preprocessador per variables categòriques
│   └── scaler.pck                # Preprocessador per variables numèriques
│
├── app.py                        # Servidor Flask (API REST)
├── client.py                     # Client per fer peticions
├── environment.yml               # Configuració de l'entorn Conda
└── README.md                     # Aquest fitxer
```

## 🏃 Executar el Servidor Flask

### Opció 1: Executar Directament

```bash
# Assegura't que l'entorn està activat
conda activate penguins-classification

# Executar el servidor
python app.py
```

### Opció 2: Executar amb el Python de l'Entorn

```bash
# Si tens problemes amb l'activació de conda
/Users/jcortes/anaconda3/envs/penguins-classification/bin/python app.py
```

El servidor s'iniciarà a **http://localhost:5001**

> ⚠️ **Nota**: El servidor utilitza el port 5001 en lloc del 5000 per defecte per evitar possibles conflictes amb altres serveis del sistema.

## 📡 API Endpoints

### 1. Health Check

Comprova que el servidor està funcionant correctament.

```bash
curl http://localhost:5001/health
```

**Resposta:**
```json
{
  "status": "ok",
  "models_loaded": 4,
  "preprocessors_loaded": 2
}
```

### 2. Llistar Models Disponibles

```bash
curl http://localhost:5001/models
```

**Resposta:**
```json
{
  "models": [
    "logistic_regression",
    "svm",
    "decision_tree",
    "knn"
  ]
}
```

### 3. Fer una Predicció

Endpoint: `POST /predict/<model_name>`

**Exemple amb Regressió Logística:**

```bash
curl -X POST http://localhost:5001/predict/logistic_regression \
  -H "Content-Type: application/json" \
  -d '{
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "Male"
  }'
```

**Resposta:**
```json
{
  "model": "logistic_regression",
  "prediction": "Adelie",
  "input": {
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "Male"
  }
}
```

### 4. Informació de l'API

```bash
curl http://localhost:5001/
```

## 🧪 Executar el Client de Prova

El projecte inclou un client que fa peticions automàtiques als 4 models:

```bash
# En una altra terminal (amb el servidor executant-se)
conda activate penguins-classification
python client.py
```

## 🔧 Desenvolupament

### Entrenar els Models

Si vols reentrenar els models, obre el notebook Jupyter:

```bash
jupyter notebook notebooks/penguins_classification.ipynb
```

El notebook conté:
- Exploració de dades (EDA)
- Preprocessament
- Entrenament dels 4 models
- Avaluació i serialització

### Exportar l'Entorn

Per crear un fitxer `environment.yml` actualitzat:

```bash
conda env export > environment.yml
```

## 📊 Exemples de Predicció

### Pingüí Adelie (Torgersen)
```json
{
  "island": "Torgersen",
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0,
  "sex": "Male"
}
```

### Pingüí Gentoo (Biscoe)
```json
{
  "island": "Biscoe",
  "bill_length_mm": 48.7,
  "bill_depth_mm": 15.1,
  "flipper_length_mm": 222.0,
  "body_mass_g": 5350.0,
  "sex": "Female"
}
```

### Pingüí Chinstrap (Dream)
```json
{
  "island": "Dream",
  "bill_length_mm": 46.5,
  "bill_depth_mm": 17.9,
  "flipper_length_mm": 192.0,
  "body_mass_g": 3500.0,
  "sex": "Female"
}
```

## ❌ Gestió d'Errors

### Model No Trobat (404)
```bash
curl -X POST http://localhost:5001/predict/invalid_model \
  -H "Content-Type: application/json" \
  -d '{"island": "Torgersen"}'
```

**Resposta:**
```json
{
  "error": "Model \"invalid_model\" no trobat. Models disponibles: ['logistic_regression', 'svm', 'decision_tree', 'knn']"
}
```

### Dades Invàlides (400)
```bash
curl -X POST http://localhost:5001/predict/logistic_regression \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'
```

**Resposta:**
```json
{
  "error": "Falten les següents columnes: ['island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']"
}
```

## 🐛 Resolució de Problemes

### Error: "No module named 'flask'"

Assegura't que l'entorn conda està activat:
```bash
conda activate penguins-classification
```

### Error: Port 5000 ja està en ús

El servidor utilitza el port 5001. Si vols canviar-lo, edita `app.py`:
```python
app.run(debug=True, port=XXXX)  # Canvia XXXX pel port desitjat
```

### Models no es carreguen

Verifica que la carpeta `models/` conté els 6 fitxers `.pck`:
```bash
ls -la models/
```

## 📝 Requisits del Projecte

- ✅ Entorn Conda configurat
- ✅ 4 models de ML entrenats i serialitzats
- ✅ API REST amb Flask
- ✅ Preprocessament de dades (DictVectorizer, StandardScaler)
- ✅ Client per fer peticions
- ✅ Gestió d'errors
- ✅ Documentació completa


## 📄 Llicència

Aquest projecte està sota llicència MIT. Consulta el fitxer `LICENSE` per més detalls.

## 🔗 Enllaços

- **Repositori GitHub:** https://github.com/jcortesa/IABD_25-26_CE_5073-3.1-tasca-avaluable
- **Dataset Palmer Penguins:** https://github.com/allisonhorst/palmerpenguins
