"""
MLP para sistema de recomendação de veículos.

Aprende a rankear veículos baseado em preferências do usuário e características dos carros.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Optional, Tuple


class RecommendationScorer:
    """MLP para scoring de veículos baseado em preferências."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(__file__).parent / "models" / "recommendation_scorer.pkl"
        self.scaler = StandardScaler()
        self.mlp: Optional[MLPRegressor] = None
        
        if self.model_path.exists():
            self.load_model()
    
    def _create_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria dados sintéticos de treinamento simulando preferências e scores.
        
        Na prática, isso seria histórico real de interações do usuário
        (cliques, visualizações, compras).
        """
        features_list = []
        scores_list = []
        
        # Simular diferentes cenários de preferência
        scenarios = [
            # Cenário 1: Busca por economia
            {
                'wants_economical': 1,
                'budget_range': 30000,
                'preferred_type_sedan': 1,
                'weight_consumo': 0.8,
                'weight_preco': 0.9,
                'weight_ano': 0.3
            },
            # Cenário 2: Busca por SUV recente
            {
                'wants_economical': 0,
                'budget_range': 80000,
                'preferred_type_suv': 1,
                'weight_consumo': 0.2,
                'weight_preco': 0.5,
                'weight_ano': 0.9
            },
            # Cenário 3: Hatch barato
            {
                'wants_economical': 1,
                'budget_range': 20000,
                'preferred_type_hatchback': 1,
                'weight_consumo': 0.7,
                'weight_preco': 1.0,
                'weight_ano': 0.1
            },
            # Cenário 4: Sedan confiável
            {
                'wants_economical': 0,
                'budget_range': 50000,
                'preferred_type_sedan': 1,
                'weight_consumo': 0.4,
                'weight_preco': 0.6,
                'weight_ano': 0.5
            },
        ]
        
        for scenario in scenarios:
            for _, vehicle in df.sample(min(500, len(df))).iterrows():
                # Features do veículo
                vehicle_features = [
                    vehicle['preco_numerico'],
                    vehicle['ano'],
                    vehicle.get('consumo_medio', 10),
                    vehicle.get('confiabilidade', 3),
                    1 if 'sedan' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
                    1 if 'suv' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
                    1 if 'hatchback' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
                ]
                
                # Features de preferência
                preference_features = [
                    scenario['wants_economical'],
                    scenario['budget_range'],
                    scenario.get('preferred_type_sedan', 0),
                    scenario.get('preferred_type_suv', 0),
                    scenario.get('preferred_type_hatchback', 0),
                ]
                
                # Combinar features
                combined = vehicle_features + preference_features
                
                # Calcular score ideal baseado nas regras
                score = self._calculate_ideal_score(vehicle, scenario)
                
                features_list.append(combined)
                scores_list.append(score)
        
        return np.array(features_list), np.array(scores_list)
    
    def _calculate_ideal_score(self, vehicle: pd.Series, scenario: Dict) -> float:
        """Calcula score ideal para treinamento supervisionado."""
        score = 0.0
        
        # Penalizar se excede orçamento
        if vehicle['preco_numerico'] > scenario['budget_range']:
            score -= 50
        else:
            # Quanto mais barato melhor (se dentro do orçamento)
            price_ratio = vehicle['preco_numerico'] / scenario['budget_range']
            score += (1 - price_ratio) * scenario.get('weight_preco', 0.5) * 30
        
        # Bonus por economia
        if scenario['wants_economical']:
            consumo = vehicle.get('consumo_medio', 10)
            if consumo > 12:  # Bom consumo
                score += scenario.get('weight_consumo', 0.5) * 20
        
        # Bonus por ano recente
        ano_score = (vehicle['ano'] - 2000) * scenario.get('weight_ano', 0.3)
        score += ano_score
        
        # Bonus por tipo correspondente
        tipo = vehicle.get('tipo_veiculo_classificado', '').lower()
        if scenario.get('preferred_type_sedan') and 'sedan' in tipo:
            score += 25
        elif scenario.get('preferred_type_suv') and 'suv' in tipo:
            score += 25
        elif scenario.get('preferred_type_hatchback') and 'hatchback' in tipo:
            score += 25
        
        # Bonus por confiabilidade
        score += vehicle.get('confiabilidade', 3) * 5
        
        return max(0, score)  # Score não pode ser negativo
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Treina o modelo de recomendação."""
        print("🔧 Gerando dados de treinamento sintéticos...")
        X, y = self._create_training_data(df)
        
        print(f"📊 Total de exemplos: {len(X)}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar MLP Regressor
        print("🧠 Treinando MLP Regressor...")
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True
        )
        
        self.mlp.fit(X_train_scaled, y_train)
        
        # Avaliar
        y_pred = self.mlp.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ MSE: {mse:.2f}")
        print(f"✅ R² Score: {r2:.2%}")
        
        return mse, r2
    
    def score_vehicles(
        self, 
        vehicles_df: pd.DataFrame,
        preferences: Dict
    ) -> np.ndarray:
        """
        Calcula scores para uma lista de veículos baseado em preferências.
        
        Args:
            vehicles_df: DataFrame com veículos a serem rankeados
            preferences: Dict com preferências do usuário
        
        Returns:
            Array de scores (um por veículo)
        """
        if self.mlp is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        features_list = []
        
        for _, vehicle in vehicles_df.iterrows():
            # Features do veículo
            vehicle_features = [
                vehicle['preco_numerico'],
                vehicle['ano'],
                vehicle.get('consumo_medio', 10),
                vehicle.get('confiabilidade', 3),
                1 if 'sedan' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
                1 if 'suv' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
                1 if 'hatchback' in vehicle.get('tipo_veiculo_classificado', '').lower() else 0,
            ]
            
            # Features de preferência
            preference_features = [
                1 if preferences.get('wants_economical') else 0,
                preferences.get('budget_max', 50000),
                1 if 'Sedan' in preferences.get('vehicle_types', []) else 0,
                1 if 'SUV' in preferences.get('vehicle_types', []) else 0,
                1 if 'Hatchback' in preferences.get('vehicle_types', []) else 0,
            ]
            
            combined = vehicle_features + preference_features
            features_list.append(combined)
        
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        scores = self.mlp.predict(X_scaled)
        return scores
    
    def save_model(self):
        """Salva o modelo treinado."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'mlp': self.mlp,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"💾 Modelo salvo em {self.model_path}")
    
    def load_model(self):
        """Carrega modelo previamente treinado."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        
        model_data = joblib.load(self.model_path)
        self.mlp = model_data['mlp']
        self.scaler = model_data['scaler']
        
        print(f"✅ Modelo carregado de {self.model_path}")


def train_recommendation_scorer(dataset_path: Optional[str] = None):
    """Script para treinar o scorer de recomendação."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    
    from tools.dataset_loader import get_dataset
    
    print("📚 Carregando dataset...")
    df = get_dataset(dataset_path).data
    
    print(f"📊 Total de veículos: {len(df)}")
    
    # Treinar
    scorer = RecommendationScorer()
    scorer.train(df)
    
    # Salvar
    scorer.save_model()
    
    return scorer


if __name__ == "__main__":
    train_recommendation_scorer()
