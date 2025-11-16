#!/usr/bin/env python3
import sys
sys.path.append('/app/server')

from ml.vehicle_classifier import VehicleTypeClassifier
import pandas as pd

vc = VehicleTypeClassifier()
vc.load_model()

test_models = [
    'Mercedes C-350 Sportcoupe 3.5',
    'Jeep Wrangler Sport V6',
    'Audi A5 3.2 FSI',
    'BMW X1 xDrive',
    'Honda Civic Si 2.0',
    'Toyota Corolla Sedan'
]

df_test = pd.DataFrame({
    'marca': ['Mercedes-Benz', 'Jeep', 'Audi', 'BMW', 'Honda', 'Toyota'],
    'modelo': test_models,
    'preco_numerico': [90000]*6,
    'ano': [2010]*6
})

preds = vc.predict(df_test)

print("\n=== TESTE DO CLASSIFIER ===")
for m, p in zip(test_models, preds):
    print(f'{m:40} -> {p}')
