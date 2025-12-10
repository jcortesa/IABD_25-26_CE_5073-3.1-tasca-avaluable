from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Carregar models i preprocessadors
def load_pickle(path):
    """Carrega un fitxer pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)

# Carregar els 4 models
models = {
    'logistic_regression': load_pickle('models/logistic_regression.pck'),
    'svm': load_pickle('models/svm.pck'),
    'decision_tree': load_pickle('models/decision_tree.pck'),
    'knn': load_pickle('models/knn.pck')
}

# Carregar preprocessadors
dv = load_pickle('models/dict_vectorizer.pck')
scaler = load_pickle('models/scaler.pck')

# Definir columnes categòriques i numèriques
CAT_COLS = ['island', 'sex']
NUM_COLS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

def preprocess_input(data):
    """
    Preprocessa les dades d'entrada aplicant DictVectorizer i StandardScaler.
    
    Args:
        data (dict): Diccionari amb les característiques del pingüí
        
    Returns:
        numpy.ndarray: Array amb les features preprocessades
    """
    # Extreure i codificar variables categòriques
    cat_data = {k: data[k] for k in CAT_COLS}
    X_cat = dv.transform([cat_data])
    
    # Extreure i normalitzar variables numèriques
    num_data = [[data[k] for k in NUM_COLS]]
    X_num = scaler.transform(num_data)
    
    # Combinar features categòriques i numèriques
    return np.hstack([X_cat, X_num])

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """
    Endpoint de predicció per a cada model.
    
    Args:
        model_name (str): Nom del model a utilitzar
        
    Returns:
        JSON amb la predicció o error
    """
    # Validar que el model existeix
    if model_name not in models:
        return jsonify({
            'error': f'Model "{model_name}" no trobat. Models disponibles: {list(models.keys())}'
        }), 404
    
    try:
        # Obtenir dades del request
        data = request.get_json()
        
        # Validar que tenim totes les columnes necessàries
        missing_cols = set(CAT_COLS + NUM_COLS) - set(data.keys())
        if missing_cols:
            return jsonify({
                'error': f'Falten les següents columnes: {list(missing_cols)}'
            }), 400
        
        # Preprocessar dades
        X = preprocess_input(data)
        
        # Fer predicció
        prediction = models[model_name].predict(X)[0]
        
        return jsonify({
            'model': model_name,
            'prediction': prediction,
            'input': data
        })
        
    except KeyError as e:
        return jsonify({
            'error': f'Clau no vàlida: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Error en processar la petició: {str(e)}'
        }), 400

@app.route('/models', methods=['GET'])
def list_models():
    """
    Llista els models disponibles.
    
    Returns:
        JSON amb la llista de models
    """
    return jsonify({
        'models': list(models.keys())
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON amb l'estat del servidor
    """
    return jsonify({
        'status': 'ok',
        'models_loaded': len(models),
        'preprocessors_loaded': 2
    })

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint principal amb informació de l'API.
    
    Returns:
        JSON amb informació de l'API
    """
    return jsonify({
        'name': 'Penguins Classification API',
        'version': '1.0',
        'endpoints': {
            'POST /predict/<model_name>': 'Fer predicció amb un model específic',
            'GET /models': 'Llistar models disponibles',
            'GET /health': 'Comprovar estat del servidor'
        },
        'available_models': list(models.keys())
    })

if __name__ == '__main__':
    print("=" * 60)
    print("SERVIDOR FLASK - CLASSIFICACIÓ DE PINGÜINS")
    print("=" * 60)
    print(f"Models carregats: {list(models.keys())}")
    print(f"Preprocessadors carregats: DictVectorizer, StandardScaler")
    print(f"Port: 5001")
    print("=" * 60)
    app.run(debug=True, port=5001)
