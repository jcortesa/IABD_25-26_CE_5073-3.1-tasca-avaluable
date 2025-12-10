import requests
import json

BASE_URL = "http://127.0.0.1:5001"

# Dades de prova (2 pingüins diferents)
# Això generarà 2 peticions per cada model (8 peticions en total)
# NOTA: Els valors han de coincidir exactament amb els del dataset!
# - island: "Torgersen", "Biscoe", "Dream"
# - sex: "Male", "Female"
test_penguins = [
    {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "Male"
    },
    {
        "island": "Biscoe",
        "bill_length_mm": 48.7,
        "bill_depth_mm": 15.1,
        "flipper_length_mm": 222.0,
        "body_mass_g": 5350.0,
        "sex": "Female"
    }
]

models = ['logistic_regression', 'svm', 'decision_tree', 'knn']

def make_prediction(model_name, data):
    """Fa una petició de predicció al servidor"""
    url = f"{BASE_URL}/predict/{model_name}"
    response = requests.post(url, json=data)
    return response.json()

def main():
    print("="*60)
    print("CLIENT DE PREDICCIÓ - PINGÜINS DE PALMER")
    print("="*60)
    
    # Comprovar que el servidor està actiu
    try:
        health = requests.get(f"{BASE_URL}/health")
        print(f"\n✓ Servidor actiu: {health.json()}")
    except requests.exceptions.ConnectionError:
        print("✗ Error: El servidor no està actiu. Executa app.py primer.")
        return
    
    # Fer 2 peticions per cada model (8 en total)
    for i, penguin in enumerate(test_penguins, 1):
        print(f"\n{'='*60}")
        print(f"PINGÜÍ #{i}")
        print(f"Dades: {json.dumps(penguin, indent=2)}")
        print("-"*60)
        
        for model in models:
            result = make_prediction(model, penguin)
            if 'error' in result:
                print(f"  {model}: ERROR - {result['error']}")
            else:
                print(f"  {model}: {result['prediction']}")
    
    print(f"\n{'='*60}")
    print(f"Total peticions realitzades: {len(test_penguins) * len(models)}")
    print("="*60)

if __name__ == '__main__':
    main()
