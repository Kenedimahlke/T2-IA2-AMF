import pandas as pd
from typing import List, Dict


class DataMerger:
    """Unifica e limpa dados extraídos"""
    
    @staticmethod
    def merge_and_clean(vehicles: List[Dict]) -> pd.DataFrame:
        """Converte lista de dicts em DataFrame limpo"""
        df = pd.DataFrame(vehicles)
        
        # Remover duplicatas usando as colunas corretas do dataset Kaggle
        if 'codigoFipe' in df.columns and 'anoModelo' in df.columns:
            df = df.drop_duplicates(subset=['codigoFipe', 'anoModelo'])
        
        # Filtrar apenas veículos com preço válido
        df = df[df['preco_numerico'] > 0]
        
        # Ordenar por marca e modelo
        df = df.sort_values(['marca', 'modelo', 'ano'])
        
        # Selecionar colunas finais
        columns_order = [
            'marca', 'modelo', 'ano', 'km', 'preco_numerico',
            'consumo_urbano_km_l', 'consumo_estrada_km_l',
            'custo_manutencao_anual', 'confiabilidade',
            'tipo_veiculo_classificado', 'segmento',
            'codigoFipe', 'mesReferencia', 'anoReferencia'
        ]
        
        # Filtrar apenas colunas que existem
        columns_order = [col for col in columns_order if col in df.columns]
        df = df[columns_order]
        
        return df
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str = "data/processed/vehicles_final.csv"):
        """Salva DataFrame em CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Dataset final salvo em {filename}")
        print(f"Total de veículos: {len(df)}")
