import pandas as pd
import random
from typing import Dict


class EnrichmentExtractor:
    """
    Enriquece dados da FIPE com informações complementares.
    Nesta versão, usa dados simulados baseados em padrões reais.
    Em produção, integraria com APIs de sites especializados.
    """
    
    # Tabelas de referência baseadas em dados médios do mercado brasileiro
    CONSUMO_MEDIO = {
        'Sedan': {'urbano': (9.5, 12.5), 'estrada': (12.0, 15.5)},
        'Hatchback': {'urbano': (11.0, 14.0), 'estrada': (13.5, 17.0)},
        'SUV': {'urbano': (8.0, 11.0), 'estrada': (10.0, 13.5)},
        'Picape': {'urbano': (7.5, 10.0), 'estrada': (9.5, 12.5)},
        'Esportivo': {'urbano': (7.0, 9.5), 'estrada': (9.0, 12.0)},
    }
    
    CUSTO_MANUTENCAO = {
        'Popular': (1500, 2500),
        'Premium': (3000, 5000),
        'Luxo': (5000, 8000),
        'SUV': (2500, 4000),
        'Esportivo': (4000, 7000),
    }
    
    CONFIABILIDADE = {
        'Toyota': (8.5, 9.5),
        'Honda': (8.0, 9.0),
        'Volkswagen': (7.0, 8.5),
        'Fiat': (6.5, 7.5),
        'Chevrolet': (7.0, 8.0),
        'Hyundai': (7.5, 8.5),
        'Jeep': (6.0, 7.5),
        'Nissan': (7.0, 8.0),
        'Renault': (6.5, 7.5),
        'Ford': (7.0, 8.0),
    }
    
    def classify_vehicle_type(self, modelo: str) -> str:
        """Classifica tipo de veículo baseado no nome do modelo"""
        modelo_lower = modelo.lower()
        
        if any(x in modelo_lower for x in ['suv', 'tracker', 'compass', 'renegade', 'tucson', 'sportage']):
            return 'SUV'
        elif any(x in modelo_lower for x in ['corolla', 'civic', 'jetta', 'fusion', 'cruze']):
            return 'Sedan'
        elif any(x in modelo_lower for x in ['gol', 'uno', 'hb20', 'onix', 'polo']):
            return 'Hatchback'
        elif any(x in modelo_lower for x in ['hilux', 'ranger', 'amarok', 's10', 'toro']):
            return 'Picape'
        elif any(x in modelo_lower for x in ['camaro', 'mustang', 'golf gti']):
            return 'Esportivo'
        else:
            return 'Hatchback'  # padrão
    
    def classify_segment(self, preco: float, marca: str) -> str:
        """Classifica segmento do veículo"""
        marcas_premium = ['BMW', 'Mercedes', 'Audi', 'Volvo', 'Land Rover']
        
        if marca in marcas_premium:
            return 'Luxo'
        elif preco > 150000:
            return 'Premium'
        elif preco > 80000:
            return 'Médio'
        else:
            return 'Popular'
    
    def estimate_mileage(self, ano: int) -> int:
        """Estima quilometragem baseada no ano"""
        current_year = 2024
        years_old = current_year - ano
        avg_km_per_year = random.uniform(12000, 18000)
        return int(years_old * avg_km_per_year)
    
    def get_consumption(self, tipo_veiculo: str) -> Dict[str, float]:
        """Obtém consumo estimado"""
        consumo_range = self.CONSUMO_MEDIO.get(tipo_veiculo, self.CONSUMO_MEDIO['Hatchback'])
        
        return {
            'consumo_urbano': round(random.uniform(*consumo_range['urbano']), 1),
            'consumo_estrada': round(random.uniform(*consumo_range['estrada']), 1)
        }
    
    def get_maintenance_cost(self, segmento: str) -> int:
        """Obtém custo anual de manutenção"""
        cost_range = self.CUSTO_MANUTENCAO.get(segmento, self.CUSTO_MANUTENCAO['Popular'])
        return int(random.uniform(*cost_range))
    
    def get_reliability(self, marca: str) -> float:
        """Obtém índice de confiabilidade (0-10)"""
        reliability_range = self.CONFIABILIDADE.get(marca, (6.0, 8.0))
        return round(random.uniform(*reliability_range), 1)
    
    def enrich_vehicle(self, vehicle: Dict) -> Dict:
        """Enriquece um único veículo com dados complementares"""
        # Limpar preço FIPE - aceita tanto 'valor' (Kaggle) quanto 'preco_fipe' (API)
        preco = 0.0
        
        if 'valor' in vehicle and vehicle['valor']:
            # Dataset do Kaggle já tem valor numérico
            preco = float(vehicle['valor'])
        elif 'preco_fipe' in vehicle and vehicle['preco_fipe']:
            # API antiga com formato "R$ X.XXX,XX"
            preco_str = vehicle['preco_fipe']
            if isinstance(preco_str, str):
                preco = float(preco_str.replace('R$ ', '').replace('.', '').replace(',', '.'))
        
        # Extrair ano - aceita tanto 'anoModelo' (Kaggle) quanto 'ano_modelo' (API)
        ano = 2020
        ano_modelo = vehicle.get('anoModelo') or vehicle.get('ano_modelo')
        
        if ano_modelo:
            if isinstance(ano_modelo, int):
                ano = ano_modelo
            elif isinstance(ano_modelo, str):
                # Extrair apenas os primeiros 4 dígitos (ano)
                ano = int(ano_modelo.split('-')[0]) if '-' in ano_modelo else int(ano_modelo[:4])
        
        # Classificações
        tipo_veiculo = self.classify_vehicle_type(vehicle.get('modelo', ''))
        segmento = self.classify_segment(preco, vehicle.get('marca', ''))
        
        # Dados enriquecidos
        consumo = self.get_consumption(tipo_veiculo)
        
        enriched = {
            **vehicle,
            'preco_numerico': preco,
            'ano': ano,
            'km': self.estimate_mileage(ano),
            'tipo_veiculo_classificado': tipo_veiculo,
            'segmento': segmento,
            'consumo_urbano_km_l': consumo['consumo_urbano'],
            'consumo_estrada_km_l': consumo['consumo_estrada'],
            'custo_manutencao_anual': self.get_maintenance_cost(segmento),
            'confiabilidade': self.get_reliability(vehicle.get('marca', ''))
        }
        
        return enriched
