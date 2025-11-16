from __future__ import annotations

import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


class VehicleDataset:
    """Centraliza carregamento e limpeza do dataset de veículos."""

    def __init__(self, dataset_path: Optional[str | Path] = None) -> None:
        self.db_url = os.getenv("AUTOMIND_DB_URL")
        self.db_table = self._sanitize_table(os.getenv("AUTOMIND_DB_TABLE", "vehicles"))
        self.dataset_path = None if self.db_url else self._resolve_path(dataset_path)
        self.data = self._load_dataframe()

    @staticmethod
    def _default_candidates() -> list[Path]:
        base_dir = Path(__file__).resolve().parents[3]
        return [
            base_dir / "automind-data-extraction/data/processed/vehicles_final.csv",
            base_dir / "automind-mcp/data/vehicles_final.csv",
        ]

    def _resolve_path(self, explicit: Optional[str | Path]) -> Path:
        if explicit:
            candidate = Path(explicit).expanduser().resolve()
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"Dataset não encontrado em {candidate}")

        for candidate in self._default_candidates():
            if candidate.is_file():
                return candidate

        joined = "\n - ".join(str(path) for path in self._default_candidates())
        raise FileNotFoundError(
            "Não foi possível localizar o dataset. Informe o caminho manualmente ou "
            f"gere o arquivo vehicles_final.csv em um dos caminhos:\n - {joined}"
        )

    def _load_dataframe(self) -> pd.DataFrame:
        if self.db_url:
            df = self._load_from_database()
        else:
            df = pd.read_csv(self.dataset_path, encoding="utf-8", engine="python")
        return self._post_process(df)

    def _load_from_database(self) -> pd.DataFrame:
        engine = create_engine(self.db_url)
        query = text(f"SELECT * FROM {self.db_table}")
        try:
            df = pd.read_sql_query(query, engine)
        finally:
            engine.dispose()
        return df

    @staticmethod
    def _sanitize_table(table_name: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        if not set(table_name) <= allowed:
            raise ValueError("Nome de tabela contém caracteres inválidos")
        return table_name

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [
            "preco_numerico",
            "consumo_urbano_km_l",
            "consumo_estrada_km_l",
            "custo_manutencao_anual",
            "confiabilidade",
            "km",
            "ano",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        text_columns = df.select_dtypes(include=["object", "string"]).columns
        for column in text_columns:
            df[column] = df[column].apply(
                lambda value: unicodedata.normalize("NFC", value)
                if isinstance(value, str)
                else value
            )

        df = df.dropna(subset=["preco_numerico"])
        df = df[df["preco_numerico"] > 0]
        df["consumo_medio"] = df[[
            "consumo_urbano_km_l",
            "consumo_estrada_km_l",
        ]].mean(axis=1, skipna=True)
        df["idade"] = 2024 - df["ano"].clip(upper=2024)
        
        # Correções de classificação de tipo de veículo
        df = self._fix_vehicle_classifications(df)

        return df.reset_index(drop=True)
    
    @staticmethod
    def _fix_vehicle_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """Corrige classificações incorretas de tipo de veículo"""
        if "tipo_veiculo_classificado" not in df.columns or "modelo" not in df.columns:
            return df
        
        # Corrigir sedans classificados como hatchback
        # Qualquer modelo com "sedan" no nome é um sedan
        mask_sedan_keyword = df["modelo"].str.contains("sedan", case=False, na=False)
        df.loc[mask_sedan_keyword, "tipo_veiculo_classificado"] = "Sedan"
        
        # Outros modelos sedan específicos
        sedan_patterns = [
            "classic",  # Corsa Classic, Accord Classic, Mercedes Classic, etc.
            "voyage",
            "logan", 
            "prisma",
            "cobalt"
        ]
        for pattern in sedan_patterns:
            mask = df["modelo"].str.contains(pattern, case=False, na=False)
            df.loc[mask, "tipo_veiculo_classificado"] = "Sedan"
        
        # Corrigir wagons/peruas classificadas como sedan
        wagon_patterns = [
            "touring",  # Honda City/Civic/Accord Touring
            "fielder",  # Toyota Corolla Fielder
            "variant",  # VW Jetta/Passat Variant
            r"\bsw\b",  # SW (Station Wagon) - word boundary para não pegar "new"
            "perua",
            "wagon",
            "sport?wagon"  # Sportwagon
        ]
        for pattern in wagon_patterns:
            mask = df["modelo"].str.contains(pattern, case=False, na=False, regex=True)
            df.loc[mask, "tipo_veiculo_classificado"] = "Wagon"
        
        return df

    @property
    def stats(self) -> dict:
        df = self.data
        return {
            "total": len(df),
            "marcas": df["marca"].nunique(),
            "modelos": df["modelo"].nunique(),
            "min_preco": float(df["preco_numerico"].min()),
            "max_preco": float(df["preco_numerico"].max()),
        }


@lru_cache(maxsize=1)
def get_dataset(path: Optional[str | Path] = None) -> VehicleDataset:
    """Carrega (com cache) uma instância compartilhada do dataset."""
    return VehicleDataset(path)


def reload_dataset(path: Optional[str | Path] = None) -> VehicleDataset:
    """Recarrega o dataset limpando o cache."""
    get_dataset.cache_clear()
    return get_dataset(path)
